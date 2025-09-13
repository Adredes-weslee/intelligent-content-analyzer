"""Hybrid retrieval utilities (BM25 + dense + RRF, union of candidates).

This module augments baseline BM25 by:
- Running BM25 for a robust lexical baseline.
- Computing dense similarities via FAISS if available; otherwise fallback
  to in-memory cosine/deterministic proxy.
- Taking the union of top‑N BM25 hits and top‑N dense hits, then combining
  ranks via Reciprocal Rank Fusion (RRF) for recall robustness.
- Returning a RetrieveResponse with diagnostics (ranks and scores).

The API Gateway calls hybrid_search to retrieve top‑k context for answer
generation.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from services.embeddings.app.embeddings import embed_texts
from shared.models import RetrieveRequest, RetrieveResponse, RetrieveResult
from shared.settings import Settings
from shared.tracing import log_event, span

from .faiss_store import search as faiss_search
from .main import INDEX as _INDEX
from .main import search as bm25_search


def _cosine_sim(a: Dict[str, int], b: Dict[str, int]) -> float:
    """Compute cosine similarity between two sparse bag-of-words dicts."""
    if not a or not b:
        return 0.0
    num = sum(a.get(k, 0) * b.get(k, 0) for k in set(a.keys()) | set(b.keys()))
    da = sum(v * v for v in a.values()) ** 0.5 or 1.0
    db = sum(v * v for v in b.values()) ** 0.5 or 1.0
    return num / (da * db)


async def hybrid_search(request: RetrieveRequest) -> RetrieveResponse:
    """Perform a hybrid search combining keyword and dense retrieval."""
    s = Settings()

    with span(
        "retrieval.hybrid",
        query=request.query,
        top_k=request.top_k,
        corr=getattr(request, "correlation_id", None),
    ):
        # 1) BM25 baseline
        bm25_resp = await bm25_search(request)
    bm25_hits = bm25_resp.hits
    bm25_ids = [h.chunk.id for h in bm25_hits]

    # Helper: chunk lookup
    id2chunk = {c.id: c for c in _INDEX}

    # 2) Dense ranking
    dense_scores: Dict[str, float] = {}
    used_faiss = False
    dense_results: List[Tuple[str, float]] = []
    try:
        # Embed query once
        qv = embed_texts([request.query])[0]
        # Candidate size for dense search
        dense_k = max(s.dense_candidates, request.top_k * 3)
        faiss_out = faiss_search(qv, top_k=dense_k)
        if faiss_out:
            used_faiss = True
            metric = (s.faiss_metric or "ip").lower()
            for cid, dist in faiss_out:
                score = float(-dist) if metric == "l2" else float(dist)
                dense_results.append((cid, score))
        else:
            raise RuntimeError("faiss returned no results")
    except Exception:
        # Fallback: BoW-cosine dense proxy over the whole index
        q_tokens = [t for t in request.query.lower().split() if t]
        q_bow: Dict[str, int] = {}
        for t in q_tokens:
            q_bow[t] = q_bow.get(t, 0) + 1
        for c in _INDEX:
            c_tokens = [t for t in c.text.lower().split() if t]
            c_bow: Dict[str, int] = {}
            for t in c_tokens:
                c_bow[t] = c_bow.get(t, 0) + 1
            dense_results.append((c.id, _cosine_sim(q_bow, c_bow)))

    # Keep top‑N dense candidates
    dense_results.sort(key=lambda kv: kv[1], reverse=True)
    top_dense = dense_results[: max(s.dense_candidates, request.top_k * 3)]
    dense_scores.update({cid: score for cid, score in top_dense})
    dense_ids = [cid for cid, _ in top_dense]

    # 3) Union of candidates
    union_ids = set(bm25_ids) | set(dense_ids)

    # 4) RRF combine
    bm25_rank: Dict[str, int] = {cid: i + 1 for i, cid in enumerate(bm25_ids)}
    dense_rank: Dict[str, int] = {cid: i + 1 for i, (cid, _) in enumerate(top_dense)}
    k = 60.0
    rrf_scores: Dict[str, float] = {}
    for cid in union_ids:
        r1 = bm25_rank.get(cid, 10**6)
        r2 = dense_rank.get(cid, 10**6)
        rrf_scores[cid] = 1.0 / (k + r1) + 1.0 / (k + r2)

    # Compose results with available bm25/dense scores
    combined: List[RetrieveResult] = []
    bm25_score_map: Dict[str, float] = {
        h.chunk.id: float(h.bm25 or 0.0) for h in bm25_hits
    }
    for cid in union_ids:
        chunk = id2chunk.get(cid)
        if not chunk:
            continue
        combined.append(
            RetrieveResult(
                chunk=chunk,
                score=rrf_scores.get(cid, 0.0),
                bm25=bm25_score_map.get(cid, 0.0),
                dense=dense_scores.get(cid, 0.0),
            )
        )

    combined.sort(key=lambda r: r.score, reverse=True)
    diag = {
        "bm25_rank": bm25_rank,
        "dense_rank": dense_rank,
        "rrf_scores": {cid: rrf_scores.get(cid, 0.0) for cid in union_ids},
        "bm25_scores": bm25_score_map,
        "dense_scores": dense_scores,
        "top_k": request.top_k,
        "dense_backend": "faiss" if used_faiss else "in_memory",
        "union_size": len(union_ids),
    }
    resp = RetrieveResponse(hits=combined[: request.top_k], diagnostics=diag)
    # Emit structured event
    log_event(
        "Retrieval",
        payload={
            "query": request.query,
            "top_ids": [h.chunk.id for h in resp.hits],
            "scores": [h.score for h in resp.hits],
            "union_size": len(union_ids),
            "dense_backend": diag["dense_backend"],
        },
        correlation_id=getattr(request, "correlation_id", None),
    )
    return resp
