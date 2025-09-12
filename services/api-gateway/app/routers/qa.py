"""Question answering router for the API gateway.

This endpoint accepts a user question, retrieves relevant chunks from
the in‑memory index and composes a naive answer by concatenating the
text of the top hits. It then evaluates the answer using a simple
heuristic and computes a confidence score. The response contains the
answer, citations and diagnostics.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from shared.models import QARequest, QAResponse, Citation, RetrieveRequest
from services.retrieval.app.main import search as retrieval_search
from services.retrieval.app.rerank import rerank
from services.evaluation.app.metrics import simple_metric_scores
from services.evaluation.app.confidence import compute_confidence

router = APIRouter()


@router.post("/ask_question", response_model=QAResponse)
async def ask_question(request: QARequest) -> QAResponse:
    """Answer a user's question using indexed document content.

    This implementation performs the following steps:
      1. Retrieve top‑k chunks from the index based on word overlap.
      2. Optionally rerank the results (no‑op in this stub).
      3. Compose an answer by concatenating the retrieved texts.
      4. Evaluate the answer using naive heuristics.
      5. Compute a confidence score from retrieval and evaluation.

    A real system would delegate retrieval and reranking to dedicated
    services and use an LLM to generate the answer.
    """
    # Step 1: retrieval
    retrieve_req = RetrieveRequest(query=request.question, top_k=request.k, hybrid=True)
    retrieve_resp = await retrieval_search(retrieve_req)
    hits = retrieve_resp.hits
    if not hits:
        # No context available; return fallback answer
        return QAResponse(
            answer="I'm sorry, I couldn't find relevant information to answer your question.",
            citations=[],
            confidence=0.0,
            diagnostics={"reason": "no_hits"},
        )
    # Step 2: rerank (no change for now)
    if request.use_rerank:
        hits = rerank(hits, model=request.reranker)
    # Step 3: compose answer by concatenation
    answer_text = " ".join(hit.chunk.text for hit in hits[: request.k])
    # Step 4: evaluation heuristic
    eval_scores = {}
    f, r, c = simple_metric_scores(request.question, answer_text, [hit.chunk for hit in hits])
    eval_scores["factuality"] = f
    eval_scores["relevance"] = r
    eval_scores["completeness"] = c
    # Step 5: compute confidence
    confidence = compute_confidence(hits, eval_scores)
    # Build citations list (just use doc_id, page and section if available)
    citations: list[Citation] = []
    for hit in hits[: request.k]:
        citations.append(
            Citation(
                doc_id=hit.chunk.doc_id,
                page=hit.chunk.meta.page,
                section=hit.chunk.meta.section,
            )
        )
    return QAResponse(
        answer=answer_text,
        citations=citations,
        confidence=confidence,
        diagnostics={"eval": eval_scores},
    )