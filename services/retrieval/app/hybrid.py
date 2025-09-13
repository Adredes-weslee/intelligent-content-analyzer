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

import re
from collections import OrderedDict
from typing import Dict, List, Tuple

from services.embeddings.app.embeddings import embed_texts
from shared.models import RetrieveRequest, RetrieveResponse, RetrieveResult
from shared.settings import Settings
from shared.tracing import log_event, span

from . import main as _ret
from .faiss_store import get_index_dim
from .faiss_store import search as faiss_search
from .main import search as bm25_search


def _tokens(s: str) -> List[str]:
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return [t for t in s.split() if t]


def _ensure_corpus_loaded() -> None:
    # Load chunks from store if INDEX is empty (in-process gateway run)
    try:
        if getattr(_ret, "INDEX", None) is not None and len(_ret.INDEX) == 0:
            _ret._load_chunks_from_store()
    except Exception:
        pass


async def hybrid_search(request: RetrieveRequest) -> RetrieveResponse:
    """Perform a hybrid search combining keyword and dense retrieval."""
    s = Settings()

    _ensure_corpus_loaded()
    with span(
        "retrieval.hybrid",
        query=request.query,
        top_k=request.top_k,
        corr=getattr(request, "correlation_id", None),
    ):
        # 1) BM25 baseline (uses main.search which now normalizes tokens)
        bm25_resp = await bm25_search(request)
    bm25_hits = bm25_resp.hits
    # Apply per-query filters to bm25 hits
    bm25_hits = [
        h
        for h in bm25_hits
        if _ret._allow_chunk(getattr(request, "filters", None), h.chunk)
    ]
    bm25_ids = [h.chunk.id for h in bm25_hits]

    # Helper: chunk lookup (filtered)
    id2chunk = {
        c.id: c
        for c in _ret.INDEX
        if _ret._allow_chunk(getattr(request, "filters", None), c)
    }

    # 2) Dense ranking
    dense_scores: Dict[str, float] = {}
    used_faiss = False
    dense_results: List[Tuple[str, float]] = []
    try:
        # Embed query once
        qv = embed_texts([request.query], dim=get_index_dim())[0]
        # Candidate size for dense search (allow per-request override)
        dense_k = max(
            (getattr(request, "dense_candidates", None) or s.dense_candidates),
            request.top_k * 3,
        )
        faiss_out = faiss_search(qv, top_k=dense_k)
        if faiss_out:
            used_faiss = True
            metric = (s.faiss_metric or "ip").lower()
            for cid, dist in faiss_out:
                # Keep only allowed chunks per filters
                if cid in id2chunk:
                    score = float(-dist) if metric == "l2" else float(dist)
                    dense_results.append((cid, score))
        else:
            raise RuntimeError("faiss returned no results")
    except Exception:
        # Fallback: BoW-cosine dense proxy over the whole filtered index
        q_tokens = _tokens(request.query)
        q_bow: Dict[str, int] = {}
        for t in q_tokens:
            q_bow[t] = q_bow.get(t, 0) + 1
        for c in id2chunk.values():
            c_tokens = _tokens(c.text)
            c_bow: Dict[str, int] = {}
            for t in c_tokens:
                c_bow[t] = c_bow.get(t, 0) + 1
            # cosine for sparse vectors
            num = sum(
                q_bow.get(k, 0) * c_bow.get(k, 0) for k in set(q_bow) | set(c_bow)
            )
            da = sum(v * v for v in q_bow.values()) ** 0.5 or 1.0
            db = sum(v * v for v in c_bow.values()) ** 0.5 or 1.0
            dense_results.append((c.id, num / (da * db)))

    # Keep top‑N dense candidates
    dense_results.sort(key=lambda kv: kv[1], reverse=True)
    top_dense = dense_results[
        : max(
            (getattr(request, "dense_candidates", None) or s.dense_candidates),
            request.top_k * 3,
        )
    ]
    dense_scores.update({cid: score for cid, score in top_dense})
    dense_ids = [cid for cid, _ in top_dense]

    # 3) Union of candidates
    union_ids = set(bm25_ids) | set(dense_ids)

    # 4) RRF combine
    bm25_rank: Dict[str, int] = {cid: i + 1 for i, cid in enumerate(bm25_ids)}
    dense_rank: Dict[str, int] = {cid: i + 1 for i, (cid, _) in enumerate(top_dense)}
    k = 20.0  # smaller k → higher RRF scores than 60
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

    # Deduplicate by chunk.id (safety; keep highest component scores)
    merged: "OrderedDict[str, RetrieveResult]" = OrderedDict()
    for r in combined:
        cid = r.chunk.id
        if cid not in merged:
            merged[cid] = r
        else:
            prev = merged[cid]
            merged[cid] = RetrieveResult(
                chunk=prev.chunk,
                score=max(prev.score, r.score),
                bm25=max(prev.bm25 or 0.0, r.bm25 or 0.0)
                if (prev.bm25 is not None or r.bm25 is not None)
                else None,
                dense=max(prev.dense or 0.0, r.dense or 0.0)
                if (prev.dense is not None or r.dense is not None)
                else None,
            )
    combined = list(merged.values())

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
        "filtered": bool(getattr(request, "filters", None)),
    }
    resp = RetrieveResponse(hits=combined[: request.top_k], diagnostics=diag)
    log_event(
        "Retrieval",
        payload={
            "query": request.query,
            "top_ids": [h.chunk.id for h in resp.hits],
            "scores": [h.score for h in resp.hits],
            "union_size": len(union_ids),
            "dense_backend": diag["dense_backend"],
            "filtered": diag["filtered"],
        },
        correlation_id=getattr(request, "correlation_id", None),
    )
    return resp
