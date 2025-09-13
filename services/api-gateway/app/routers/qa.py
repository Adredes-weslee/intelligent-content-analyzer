"""Question answering router for the API gateway.

Deterministic orchestration with caching:
- Retrieval -> (optional re-rank) -> Thresholding/Refinement ->
  Cache check (semantic+exact with content fingerprint) ->
  Generation -> Evaluation/Confidence -> Response.

Caching & Performance:
- Exact cache by normalized question (fast path).
- Semantic cache of answers using question embeddings and a
  content_fingerprint over top document IDs to avoid stale hits.
- TTLs configurable via Settings.answer_cache_ttl_seconds.
- Rate limiting supported (enabled by default: 5 req/min per normalized question).

Tracing spans and diagnostics are emitted for each step.
"""

from __future__ import annotations

import os
import string
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException

from services.llm_generate.app.main import generate_answer as llm_generate
from services.llm_generate.app.main import summarize_document as llm_summarize
from services.llm_generate.app.prompts import RETRIEVER_REWRITER_PROMPT
from services.retrieval.app.hybrid import hybrid_search as retrieval_search
from services.retrieval.app.rerank import rerank
from shared.cache import (
    content_fingerprint,
    get_default_cache,
    get_semantic_cache,
    semantic_key,
)
from shared.models import (
    Citation,
    EvaluateRequest,
    FeedbackAck,
    FeedbackRequest,
    GenerateRequest,
    QARequest,
    QAResponse,
    RetrieveRequest,
    SummarizeRequest,
)
from shared.settings import Settings
from shared.tracing import log_event, span

# Optional Gemini client for query refinement
_genai = None
_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if _GEMINI_API_KEY:
    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=_GEMINI_API_KEY)
        _genai = genai
    except Exception:
        _genai = None

router = APIRouter()
_cache = get_default_cache()
_settings = Settings()


def _canonicalize_query(q: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return " ".join(q.lower().translate(table).split())


async def _refine_query(original: str) -> Optional[str]:
    """Return a refined query using an LLM prompt, or None on failure."""
    if not _settings.query_refine_enabled or _genai is None:
        return None
    try:
        model = _genai.GenerativeModel(_settings.gemini_fast_model)
        resp = model.generate_content(
            [
                {"role": "system", "parts": [{"text": RETRIEVER_REWRITER_PROMPT}]},
                {"role": "user", "parts": [{"text": original}]},
            ]
        )
        text = getattr(resp, "text", "") or ""
        # Heuristic: pick the first non-empty line as the refined query
        for line in text.splitlines():
            candidate = line.strip(" -•\t").strip()
            if candidate and len(candidate) > 3:
                return candidate
    except Exception:
        pass
    return None


@router.post("/ask_question", response_model=QAResponse)
async def ask_question(request: QARequest) -> QAResponse:
    """Answer a user's question using indexed document content."""
    # Generate a correlation id for observability across services
    correlation_id = str(uuid.uuid4())

    # Basic rate limit per normalized question
    if _settings.rate_limit_enabled:
        norm = _canonicalize_query(request.question)
        rl_key = f"rl:qpm:{norm}"
        try:
            rec = _cache.get(rl_key) or {"count": 0}
            cnt = int(rec.get("count", 0)) + 1
            rec["count"] = cnt
            _cache.set(rl_key, rec, ttl=60)
            if cnt > _settings.rate_limit_per_minute:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
        except HTTPException:
            raise
        except Exception:
            pass

    # Cache lookup
    exact_key = f"qa:{_canonicalize_query(request.question)}:k={request.k}:rr={request.use_rerank}"
    sem_key = semantic_key(request.question)
    cached = None
    if _settings.cache_enabled:
        cached = _cache.get(exact_key) or _cache.get(sem_key)
    if cached:
        return QAResponse(**cached)

    # Step 1: retrieval
    with span(
        "qa.retrieve", k=request.k, rerank=request.use_rerank, corr=correlation_id
    ):
        retrieve_req = RetrieveRequest(
            query=request.question,
            top_k=max(request.k, 10),
            hybrid=True,
            correlation_id=correlation_id,
        )
        retrieve_resp = await retrieval_search(retrieve_req)
        hits = retrieve_resp.hits

    if not hits:
        return QAResponse(
            answer="I'm sorry, I couldn't find relevant information to answer your question.",
            citations=[],
            confidence=0.0,
            diagnostics={"reason": "no_hits", "correlation_id": correlation_id},
        )

    # Step 2: rerank
    if request.use_rerank:
        with span("qa.rerank"):
            hits = rerank(
                hits,
                model=_settings.reranker_model
                if _settings.reranker_backend == "cross-encoder"
                else None,
                query=request.question,
            )

    # Thresholding on retrieval quality
    top_score = float(hits[0].score if hits else 0.0)
    if top_score < _settings.rerank_threshold:
        refined = await _refine_query(request.question)
        if refined:
            with span("qa.retrieve_refined", refined=True, corr=correlation_id):
                retrieve_req2 = RetrieveRequest(
                    query=refined,
                    top_k=max(request.k, 10),
                    hybrid=True,
                    correlation_id=correlation_id,
                )
                retrieve_resp2 = await retrieval_search(retrieve_req2)
                hits2 = retrieve_resp2.hits
            if hits2:
                if request.use_rerank:
                    with span("qa.rerank_refined"):
                        hits2 = rerank(
                            hits2,
                            model=_settings.reranker_model
                            if _settings.reranker_backend == "cross-encoder"
                            else None,
                            query=refined,
                        )
                if hits2 and float(hits2[0].score) >= _settings.rerank_threshold:
                    hits = hits2
                    top_score = float(hits[0].score)

    if top_score < _settings.rerank_threshold:
        # Abstain early on low retrieval confidence
        resp = QAResponse(
            answer="Sorry, I couldn’t find relevant information to answer confidently.",
            citations=[],
            confidence=0.0,
            diagnostics={
                "reason": "low_retrieval_confidence",
                "top_score": round(top_score, 4),
                "threshold": _settings.rerank_threshold,
                "retrieval": getattr(retrieve_resp, "diagnostics", None),
                "correlation_id": correlation_id,
            },
        )
        if _settings.cache_enabled:
            _cache.set(exact_key, resp.dict(), ttl=600)
            _cache.set(sem_key, resp.dict(), ttl=600)
        return resp

    # Step 3: assemble structured context and call generator
    with span("qa.generate", corr=correlation_id):
        top_chunks = [h.chunk for h in hits[: request.k]]
        gen_req = GenerateRequest(
            question=request.question,
            context_chunks=top_chunks,
            correlation_id=correlation_id,
        )
        gen_resp = await llm_generate(gen_req)  # returns QAResponse
        answer_text = gen_resp.answer

    # Step 4: evaluation (service call with extended metrics)
    with span("qa.evaluate", corr=correlation_id):
        # Call local service module directly to avoid HTTP hop
        from services.evaluation.app.main import (
            evaluate as eval_endpoint,  # local import
        )

        ereq = EvaluateRequest(
            question=request.question,
            answer=answer_text,
            sources=[hit.chunk for hit in hits],
            hits=hits,
        )
        eresp = await eval_endpoint(ereq)
        eval_scores = {
            "factuality": eresp.factuality,
            "relevance": eresp.relevance,
            "completeness": eresp.completeness,
            # expose extended metrics for diagnostics/confidence tuning
            "faithfulness": eresp.faithfulness
            if eresp.faithfulness is not None
            else 0.0,
            "answer_relevance_1_5": eresp.answer_relevance_1_5
            if eresp.answer_relevance_1_5 is not None
            else 0.0,
            "context_relevance_ratio": eresp.context_relevance_ratio
            if eresp.context_relevance_ratio is not None
            else 0.0,
        }
        # Emit evaluation event for observability
        log_event("Evaluation", payload=eval_scores, correlation_id=correlation_id)

    # Step 5: compute confidence
    from services.evaluation.app.confidence import compute_confidence  # local import

    confidence = compute_confidence(hits, eval_scores)

    if confidence < _settings.confidence_threshold:
        resp = QAResponse(
            answer="I’m not confident enough to answer based on the available context.",
            citations=[],
            confidence=confidence,
            diagnostics={
                "eval": eval_scores,
                "reason": "low_confidence",
                "threshold": _settings.confidence_threshold,
                "correlation_id": correlation_id,
            },
        )
        if _settings.cache_enabled:
            _cache.set(exact_key, resp.dict(), ttl=600)
            _cache.set(sem_key, resp.dict(), ttl=600)
        return resp

    # Build citations (prefer generator; fallback to top chunks)
    citations: list[Citation] = (
        gen_resp.citations
        if gen_resp.citations
        else [
            Citation(
                doc_id=h.chunk.doc_id,
                page=h.chunk.meta.page,
                section=h.chunk.meta.section,
            )
            for h in hits[: request.k]
        ]
    )

    # Build fingerprint and semantic cache key
    fp = content_fingerprint([h.chunk.doc_id for h in hits[: max(10, request.k)]])
    fp_key = f"qa2:{_canonicalize_query(request.question)}:fp={fp}:k={request.k}:rr={request.use_rerank}"
    sem_cache = get_semantic_cache()

    resp = QAResponse(
        answer=answer_text,
        citations=citations,
        confidence=confidence,
        diagnostics={
            "eval": eval_scores,
            "retrieval": {
                "num_hits": len(hits),
                "top_scores": [round(h.score, 4) for h in hits[: request.k]],
                "diagnostics": getattr(retrieve_resp, "diagnostics", None),
            },
            "correlation_id": correlation_id,
            "fingerprint": fp,
        },
    )
    if _settings.cache_enabled:
        ttl = _settings.answer_cache_ttl_seconds
        try:
            _cache.set(exact_key, resp.dict(), ttl=ttl)
            _cache.set(sem_key, resp.dict(), ttl=ttl)
            _cache.set(fp_key, resp.dict(), ttl=ttl)
            sem_cache.add(request.question, fp, fp_key)
        except Exception:
            pass
    return resp


@router.get("/document_summary")
async def document_summary(doc_id: str, max_chunks: int | None = None) -> dict:
    """Summarize a document by doc_id using the LLM summarizer.

    Collects up to `max_chunks` chunks for the given document and calls
    the Gemini-based summarizer. Falls back to a deterministic summary
    when offline.
    """
    # Lazy import to avoid circulars and keep retrieval concerns isolated
    from services.retrieval.app.main import INDEX as RETRIEVAL_INDEX  # local import

    limit = max_chunks or _settings.summarizer_max_chunks
    correlation_id = str(uuid.uuid4())
    with span("summary.collect_chunks", doc_id=doc_id, max_chunks=limit):
        chunks = [c for c in RETRIEVAL_INDEX if getattr(c, "doc_id", None) == doc_id]
        if not chunks:
            return {
                "summary": "No content available for the specified document.",
                "key_points": [],
                "citations": [],
                "diagnostics": {
                    "reason": "no_chunks",
                    "correlation_id": correlation_id,
                },
            }
        chunks = chunks[:limit]

    with span("summary.llm"):
        sreq = SummarizeRequest(
            doc_id=doc_id, correlation_id=correlation_id, chunks=chunks
        )  # type: ignore[arg-type]
        sresp = await llm_summarize(sreq)
        return {
            "summary": sresp.summary,
            "key_points": sresp.key_points,
            "citations": [c.dict() for c in sresp.citations],
            "diagnostics": sresp.diagnostics,
        }


@router.post("/feedback", response_model=FeedbackAck)
async def submit_feedback(request: FeedbackRequest) -> FeedbackAck:
    """Capture end-user feedback and log an observability event.

    Stores feedback in the default cache for analytics and emits a Feedback
    structured event. Feedback is keyed by a generated id and retained for a
    limited duration.
    """
    fid = str(uuid.uuid4())
    record = {
        "id": fid,
        "correlation_id": request.correlation_id,
        "question": request.question,
        "answer": request.answer,
        "rating": request.rating,
        "comment": request.comment,
    }
    try:
        if _settings.cache_enabled:
            _cache.set(f"feedback:{fid}", record, ttl=30 * 24 * 3600)
        stored = True
    except Exception:
        stored = False
    log_event(
        "Feedback",
        payload={
            "rating": request.rating,
            "has_comment": bool(request.comment),
            "question_chars": len(request.question or ""),
        },
        correlation_id=request.correlation_id,
    )
    return FeedbackAck(ok=True, stored=stored)
