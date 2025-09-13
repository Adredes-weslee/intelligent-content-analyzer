"""Retrieval microservice (BM25 + dense cosine blend; FAISS-assisted).

Overview
- Maintains an in-memory list of DocChunk objects and parallel, L2-normalized
  dense vectors for each chunk (DENSE_VECTORS, VEC_BY_CHUNK_ID).
- Endpoints:
  • POST /index  – embeds incoming chunks and updates in-memory stores and the
                   optional FAISS index (see faiss_store.upsert_vectors).
  • POST /search – runs BM25 keyword search; when hybrid=True also computes a
                   dense similarity score and blends: score = 0.5*bm25 + 0.5*dense.

Algorithms
- BM25: lower-cased whitespace tokenization, IDF and length normalization.
- Dense: embeddings via services.embeddings.app.embeddings.embed_texts;
         vectors are L2-normalized and compared with cosine similarity.
- FAISS: if available (faiss_store.init_index() succeeded), dense lookup may be
         accelerated via FAISS; otherwise cosine is computed in Python.

Use this vs hybrid.py
- /search here is a lightweight hybrid that blends per-chunk scores.
- For a stronger hybrid that unions top-N BM25 with top-N dense candidates
  and merges via Reciprocal Rank Fusion (RRF) with diagnostics, use
  services.retrieval.app.hybrid.hybrid_search from the API Gateway.

Notes & limitations
- No metadata filters/fielded search in this module.
- In-memory only (no persistence/sharding); FAISS persistence is handled in
  services.retrieval.app.faiss_store.
- Idempotency and document bookkeeping live in the API Gateway.
"""

from __future__ import annotations

import math
from typing import Dict, List

from fastapi import FastAPI

from services.embeddings.app.embeddings import embed_texts
from shared.models import (
    DocChunk,
    RetrieveRequest,
    RetrieveResponse,
    RetrieveResult,
)

from .faiss_store import get_index_dim, init_index, upsert_vectors

app = FastAPI(title="Retrieval Service", version="0.1.0")

# In‑memory list of indexed chunks
INDEX: List[DocChunk] = []
# Parallel dense vectors and lookup by chunk id
DENSE_VECTORS: List[List[float]] = []
VEC_BY_CHUNK_ID: Dict[str, List[float]] = {}

# Initialize FAISS index (no-op if faiss not installed)
init_index()


def _l2_norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v)) or 1.0


def _normalize(v: List[float]) -> List[float]:
    n = _l2_norm(v)
    return [x / n for x in v]


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def _allow_chunk(filters, chunk) -> bool:
    if not filters:
        return True
    if getattr(filters, "include_doc_ids", None):
        if chunk.doc_id not in set(filters.include_doc_ids):
            return False
    meta = getattr(chunk, "meta", None)
    src = (
        getattr(meta, "source", None)
        if meta and not isinstance(meta, dict)
        else (meta.get("source") if isinstance(meta, dict) else None)
    )
    lang = (
        getattr(meta, "lang", None)
        if meta and not isinstance(meta, dict)
        else (meta.get("lang") if isinstance(meta, dict) else None)
    )
    if getattr(filters, "include_sources", None):
        if src not in set(filters.include_sources):
            return False
    if getattr(filters, "lang", None):
        if lang != filters.lang:
            return False
    return True


def add_chunks(chunks: List[DocChunk]) -> int:
    """Embed and add chunks to both lexical and dense indexes."""
    if not chunks:
        return 0
    texts = [c.text for c in chunks]
    # Embed with FAISS-consistent dimension
    vecs = embed_texts(texts, batch_size=16, dim=get_index_dim())
    count = 0
    # Upsert into FAISS vector store for persistence
    try:
        upsert_vectors([c.id for c in chunks], vecs)
    except Exception:
        # best-effort; continue with in-memory only
        pass
    for c, v in zip(chunks, vecs):
        INDEX.append(c)
        nv = _normalize([float(x) for x in v])
        DENSE_VECTORS.append(nv)
        VEC_BY_CHUNK_ID[c.id] = nv
        count += 1
    return count


def remove_chunks_by_ids(chunk_ids: List[str]) -> int:
    """Remove chunks from in-memory stores by their IDs.

    Returns number of removed items.
    """
    if not chunk_ids or not INDEX:
        return 0
    remove_set = set(chunk_ids)
    new_index: List[DocChunk] = []
    new_vecs: List[List[float]] = []
    removed = 0
    for c, v in zip(INDEX, DENSE_VECTORS):
        if c.id in remove_set:
            removed += 1
            VEC_BY_CHUNK_ID.pop(c.id, None)
            continue
        new_index.append(c)
        new_vecs.append(v)
    INDEX.clear()
    INDEX.extend(new_index)
    DENSE_VECTORS.clear()
    DENSE_VECTORS.extend(new_vecs)
    return removed


def _unique_hits(hits: List[RetrieveResult]) -> List[RetrieveResult]:
    """De-duplicate results by chunk.id while keeping first/highest-ranked entry."""
    seen = set()
    out: List[RetrieveResult] = []
    for r in hits:
        cid = r.chunk.id
        if cid in seen:
            continue
        seen.add(cid)
        out.append(r)
    return out


@app.post("/index")
async def index_chunks(chunks: List[dict]) -> dict:
    """Add a list of chunks to the in‑memory index.

    This endpoint accepts raw dictionaries to avoid tight coupling to
    Pydantic models across services. Each item is parsed into a
    DocChunk instance and appended to the index.
    """
    parsed: List[DocChunk] = []
    for c in chunks:
        parsed.append(DocChunk.parse_obj(c))
    count = add_chunks(parsed)
    return {"status": "ok", "indexed": count}


@app.post("/search", response_model=RetrieveResponse)
async def search(req: RetrieveRequest) -> RetrieveResponse:
    """Return a ranked list of document chunks matching the query.

    Implements BM25 keyword scoring over the in-memory index. Dense
    retrieval computes cosine similarity against stored vectors. When
    hybrid=True we blend: 0.5*bm25 + 0.5*dense.
    """
    # Tokenize query
    query_tokens = [t for t in req.query.lower().split() if t]
    candidates: List[DocChunk] = [
        c for c in INDEX if _allow_chunk(getattr(req, "filters", None), c)
    ]
    if not query_tokens or not candidates:
        return RetrieveResponse(hits=[])

    # Precompute corpus statistics over filtered candidates
    doc_terms: List[List[str]] = [c.text.lower().split() for c in candidates]
    doc_lens: List[int] = [max(1, len(ts)) for ts in doc_terms]
    avgdl = sum(doc_lens) / len(doc_lens)
    N = len(candidates)

    # BM25 parameters
    k1, b = 1.5, 0.75

    # Document frequency for each unique term in query
    unique_q = set(query_tokens)
    df = {t: sum(1 for terms in doc_terms if t in set(terms)) for t in unique_q}
    idf = {
        t: math.log((N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1.0)
        for t in unique_q
    }

    # Compute query embedding for dense retrieval (normalized), using FAISS dim
    q_vecs = embed_texts([req.query], dim=get_index_dim())
    qv = _normalize([float(x) for x in q_vecs[0]]) if q_vecs else []

    results: List[RetrieveResult] = []
    for i, chunk in enumerate(candidates):
        terms = doc_terms[i]
        dl = doc_lens[i]
        bm25_score = 0.0
        # term frequency in this document
        for t in unique_q:
            tf = terms.count(t)
            if tf == 0:
                continue
            numerator = tf * (k1 + 1.0)
            denominator = tf + k1 * (1.0 - b + b * (dl / avgdl))
            bm25_score += idf[t] * (numerator / denominator)
        # Dense cosine similarity using stored vectors
        cv = VEC_BY_CHUNK_ID.get(chunk.id)
        dense_score = _cosine(qv, cv) if cv else 0.0
        final_score = (
            bm25_score if not req.hybrid else (0.5 * bm25_score + 0.5 * dense_score)
        )
        results.append(
            RetrieveResult(
                chunk=chunk,
                score=final_score,
                bm25=bm25_score,
                dense=dense_score,
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    results = _unique_hits(results)
    top_hits = results[: req.top_k]
    diag = {
        "mode": "hybrid" if req.hybrid else "bm25",
        "top_k": req.top_k,
        "filtered": True if getattr(req, "filters", None) else False,
        "bm25_scores": {c.id: r.bm25 for c, r in zip([c for c in candidates], results)},
    }
    return RetrieveResponse(hits=top_hits, diagnostics=diag)
