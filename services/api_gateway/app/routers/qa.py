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

import string
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException

from services.llm_generate.app.main import generate_answer as llm_generate
from services.llm_generate.app.prompts import RETRIEVER_REWRITER_PROMPT
from services.retrieval.app.hybrid import hybrid_search as retrieval_search
from services.retrieval.app.rerank import rerank
from shared.cache import (
    content_fingerprint,
    get_default_cache,
    get_index_version,
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
)
from shared.settings import Settings
from shared.tracing import log_event, span

_settings = Settings()
# Optional Gemini client for query refinement and translation
_genai = None
_GEMINI_API_KEY = _settings.gemini_api_key
if _GEMINI_API_KEY:
    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=_GEMINI_API_KEY)
        _genai = genai
    except Exception:
        _genai = None

# Optional language detection (AF5)
try:
    from langdetect import detect as _langdetect_detect  # type: ignore
except Exception:
    _langdetect_detect = None

router = APIRouter()
_cache = get_default_cache()


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


async def _expand_queries(original: str) -> List[str]:
    """Return up to 3 refined variants (original + LLM rewrites)."""
    variants: List[str] = [original]
    refined = await _refine_query(original)
    if refined and refined not in variants:
        variants.append(refined)
    # Best-effort extra variants if LLM present
    if _genai is not None and _settings.query_refine_enabled:
        try:
            model = _genai.GenerativeModel(_settings.gemini_fast_model)
            resp = model.generate_content(
                [
                    {"role": "system", "parts": [{"text": RETRIEVER_REWRITER_PROMPT}]},
                    {"role": "user", "parts": [{"text": original}]},
                ]
            )
            text = getattr(resp, "text", "") or ""
            for line in text.splitlines():
                cand = line.strip(" -•\t").strip()
                if cand and cand not in variants and len(variants) < 3:
                    variants.append(cand)
        except Exception:
            pass
    return variants[:3]


def _detect_lang(text: str) -> Optional[str]:
    if not _langdetect_detect:
        return None
    try:
        return _langdetect_detect(text)
    except Exception:
        return None


async def _translate(text: str, target_lang: str) -> Optional[str]:
    """Translate text to target_lang using Gemini if available; else None."""
    if _genai is None:
        return None
    try:
        model = _genai.GenerativeModel(_settings.gemini_fast_model)
        prompt = f"Translate the following text into {target_lang}. Return only the translation."
        resp = model.generate_content(
            [
                {"role": "user", "parts": [{"text": prompt}]},
                {"role": "user", "parts": [{"text": text}]},
            ]
        )
        out = getattr(resp, "text", "") or ""
        return out.strip() or None
    except Exception:
        return None


@router.post("/ask_question", response_model=QAResponse)
async def ask_question(request: QARequest) -> QAResponse:
    """Answer a user's question using indexed document content."""
    # Generate a correlation id for observability across services
    correlation_id = str(uuid.uuid4())
    index_version = get_index_version()

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

    exact_key = f"qa:v={index_version}:{_canonicalize_query(request.question)}:k={request.k}:rr={request.use_rerank}"
    sem_key = f"{semantic_key(request.question)}:v={index_version}"
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
            filters=getattr(request, "filters", None),
            dense_candidates=getattr(request, "dense_candidates", None),
        )
        retrieve_resp = await retrieval_search(retrieve_req)
        hits = retrieve_resp.hits

    if not hits:
        resp = QAResponse(
            answer="I'm sorry, I couldn't find relevant information to answer your question.",
            citations=[],
            confidence=0.0,
            diagnostics={"reason": "no_hits", "correlation_id": correlation_id},
        )
        if _settings.cache_enabled:
            ttl = _settings.answer_cache_ttl_seconds
            _cache.set(exact_key, resp.dict(), ttl=ttl)
            _cache.set(sem_key, resp.dict(), ttl=ttl)
        return resp

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

    # AF6: Multi-variant refinement union with optional re-rank
    if top_score < _settings.rerank_threshold:
        variants = await _expand_queries(request.question)
        union_hits = []
        seen_ids = set()
        for q2 in variants:
            with span("qa.retrieve_refined", refined=True, corr=correlation_id, q=q2):
                retrieve_req2 = RetrieveRequest(
                    query=q2,
                    top_k=max(request.k, 10),
                    hybrid=True,
                    correlation_id=correlation_id,
                    filters=getattr(request, "filters", None),
                    dense_candidates=getattr(request, "dense_candidates", None),
                )
                retrieve_resp2 = await retrieval_search(retrieve_req2)
                for h in retrieve_resp2.hits or []:
                    cid = getattr(h.chunk, "id", None)
                    if cid and cid not in seen_ids:
                        union_hits.append(h)
                        seen_ids.add(cid)
        if union_hits:
            if request.use_rerank:
                with span("qa.rerank_refined"):
                    union_hits = rerank(
                        union_hits,
                        model=_settings.reranker_model
                        if _settings.reranker_backend == "cross-encoder"
                        else None,
                        query=request.question,
                    )
            else:
                union_hits.sort(key=lambda r: float(r.score or 0.0), reverse=True)
            if float(union_hits[0].score or 0.0) > top_score:
                hits = union_hits
                top_score = float(hits[0].score or 0.0)

    # AF5: Translation fallback (question -> dominant doc language) if still low
    if top_score < _settings.rerank_threshold and _genai is not None:
        q_lang = _detect_lang(request.question)
        # Majority language among top hits (by meta.lang if present)
        counts = {}
        for h in hits[: max(5, request.k)]:
            # meta may be a pydantic object or dict; support both
            meta = getattr(h.chunk, "meta", None)
            doc_lang = None
            if isinstance(meta, dict):
                doc_lang = meta.get("lang")
            else:
                doc_lang = getattr(meta, "lang", None)
            if doc_lang:
                counts[doc_lang] = counts.get(doc_lang, 0) + 1
        dom_lang = max(counts, key=counts.get) if counts else None
        if q_lang and dom_lang and q_lang != dom_lang:
            translated = await _translate(request.question, dom_lang)
            if translated:
                with span(
                    "qa.retrieve_translated",
                    src=q_lang,
                    tgt=dom_lang,
                    corr=correlation_id,
                ):
                    retrieve_req3 = RetrieveRequest(
                        query=translated,
                        top_k=max(request.k, 10),
                        hybrid=True,
                        correlation_id=correlation_id,
                        filters=getattr(request, "filters", None),
                        dense_candidates=getattr(request, "dense_candidates", None),
                    )
                    retrieve_resp3 = await retrieval_search(retrieve_req3)
                    hits3 = retrieve_resp3.hits or []
                if request.use_rerank and hits3:
                    with span("qa.rerank_translated"):
                        hits3 = rerank(
                            hits3,
                            model=_settings.reranker_model
                            if _settings.reranker_backend == "cross-encoder"
                            else None,
                            query=translated,
                        )
                if hits3 and float(hits3[0].score or 0.0) > top_score:
                    hits = hits3
                    top_score = float(hits[0].score or 0.0)

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
            ttl = _settings.answer_cache_ttl_seconds
            _cache.set(exact_key, resp.dict(), ttl=ttl)
            _cache.set(sem_key, resp.dict(), ttl=ttl)
        return resp

    # Semantic cache lookup using content fingerprint before generation
    fp = content_fingerprint([h.chunk.doc_id for h in hits[: max(10, request.k)]])
    if _settings.cache_enabled:
        with span("qa.semantic_cache_lookup", corr=correlation_id):
            try:
                sem_cache = get_semantic_cache()
                key = sem_cache.query(request.question, fp)
                if key:
                    cached_sem = _cache.get(key)
                    if cached_sem:
                        return QAResponse(**cached_sem)
            except Exception:
                # Best-effort semantic cache; continue on errors
                pass

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

    # Step 5: compute confidence (using judge_scores support)
    from services.evaluation.app.confidence import compute_confidence  # local import

    # Judge payload may not be present; derive a proxy from faithfulness and answer_relevance_1_5
    judge_scores = {
        "faithfulness": float(eval_scores.get("faithfulness", 0.0) or 0.0),
    }
    try:
        ar15 = eval_scores.get("answer_relevance_1_5")
        if ar15 is not None:
            judge_scores["answer_relevance_norm"] = float(ar15) / 5.0
    except Exception:
        pass

    # Retrieval aggregates from current hit list
    retrieval_top = float(hits[0].score or 0.0)
    used_hits = hits[
        : max(
            1,
            min(
                len(hits),
                _settings.summarizer_max_chunks
                if hasattr(_settings, "summarizer_max_chunks")
                else request.k,
            ),
        )
    ]
    retrieval_mean = float(sum(float(h.score or 0.0) for h in used_hits)) / max(
        1, len(used_hits)
    )

    confidence = compute_confidence(
        retrieval_top=retrieval_top,
        retrieval_mean=retrieval_mean,
        eval_scores={
            "factuality": float(eval_scores.get("factuality", 0.0) or 0.0),
            "relevance": float(eval_scores.get("relevance", 0.0) or 0.0),
            "completeness": float(eval_scores.get("completeness", 0.0) or 0.0),
            "faithfulness": float(eval_scores.get("faithfulness", 0.0) or 0.0),
        },
        judge_scores=judge_scores,
    )

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
            ttl = _settings.answer_cache_ttl_seconds
            _cache.set(exact_key, resp.dict(), ttl=ttl)
            _cache.set(sem_key, resp.dict(), ttl=ttl)
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

    # Build fingerprint and semantic cache key (reuse fp from lookup)
    fp_key = f"qa2:v={index_version}:{_canonicalize_query(request.question)}:fp={fp}:k={request.k}:rr={request.use_rerank}"
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
