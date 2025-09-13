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
- BM25: normalized tokenization, IDF and length normalization.
- Dense: embeddings via services.embeddings.app.embeddings.embed_texts;
         vectors are L2-normalized and compared with cosine similarity.
- FAISS: if available (faiss_store.init_index() succeeded), dense lookup may be
         accelerated via FAISS; otherwise cosine is computed in Python.

Notes & limitations
- No metadata filters/fielded search in this module.
- In-memory only (no persistence/sharding); FAISS persistence is handled in
  services.retrieval.app.faiss_store.
- Idempotency and document bookkeeping live in the API Gateway.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List

import httpx
from fastapi import FastAPI

from services.embeddings.app.embeddings import embed_texts
from shared.models import (
    DocChunk,
    RetrieveRequest,
    RetrieveResponse,
    RetrieveResult,
)
from shared.settings import Settings
from shared.tracing import install_fastapi_tracing

from .faiss_store import _DOC_MAP_PATH, get_index_dim, init_index, upsert_vectors
from .faiss_store import search as faiss_search

app = FastAPI(title="Retrieval Service", version="0.1.0")
install_fastapi_tracing(app, service_name="retrieval")


_settings = Settings()
_EMBED_URL = (os.getenv("EMBEDDINGS_URL") or _settings.embeddings_url or "").rstrip("/")


def _embed_texts_http(
    texts: list[str], dim: int | None = None, batch_size: int = 16
) -> list[list[float]]:
    if not _EMBED_URL:
        raise RuntimeError("EMBEDDINGS_URL not configured")
    payload = {"texts": texts}
    if dim is not None:
        payload["dim"] = dim
    with httpx.Client(
        timeout=httpx.Timeout(connect=5, read=60, write=10, pool=5)
    ) as client:
        r = client.post(f"{_EMBED_URL}/embed", json=payload)
        r.raise_for_status()
        return r.json().get("embeddings", [])


def embed_texts_auto(
    texts: list[str], *, dim: int | None = None, batch_size: int = 16
) -> list[list[float]]:
    # Prefer remote embeddings when configured; fall back to local model.
    if _EMBED_URL:
        try:
            return _embed_texts_http(texts, dim=dim, batch_size=batch_size)
        except Exception:
            # Fallback to local on error
            pass
    return embed_texts(texts, batch_size=batch_size, dim=dim)


@app.get("/")
def _root():
    return {"status": "ok", "service": "retrieval"}


@app.get("/health")
def _health():
    return {"status": "ok"}


INDEX: List[DocChunk] = []
DENSE_VECTORS: List[List[float]] = []
VEC_BY_CHUNK_ID: Dict[str, List[float]] = {}
CHUNK_BY_ID: Dict[str, DocChunk] = {}

init_index()


@app.get("/_status")
def _status():
    return {
        "indexed": len(INDEX),
        "doc_map_path": str(_DOC_MAP_PATH),
        "doc_map_exists": Path(_DOC_MAP_PATH).exists(),
        "sample_ids": list(CHUNK_BY_ID.keys())[:5],
    }


# NEW: public status endpoint used by the API Gateway
@app.get("/status")
def status():
    # Prefer env var; fall back to sibling of doc_map.json
    faiss_path = os.getenv("FAISS_INDEX_PATH") or str(
        Path(_DOC_MAP_PATH).with_name("faiss.index")
    )
    return {
        "indexed": len(INDEX),
        "doc_map_path": str(_DOC_MAP_PATH),
        "doc_map_exists": Path(_DOC_MAP_PATH).exists(),
        "faiss_path": faiss_path,
        "faiss_exists": Path(faiss_path).exists(),
        "sample_ids": list(CHUNK_BY_ID.keys())[:5],
    }


@app.get("/chunks_by_doc")
def chunks_by_doc(doc_id: str, max_chunks: int | None = None) -> dict:
    """Return chunks for a given doc_id (used by summarization)."""
    limit = max_chunks or 50
    items = [c for c in INDEX if getattr(c, "doc_id", None) == doc_id][:limit]
    try:
        chunks = [c.dict() for c in items]
    except Exception:
        chunks = [
            c if isinstance(c, dict) else getattr(c, "__dict__", {}) for c in items
        ]
    return {"chunks": chunks}


def _l2_norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v)) or 1.0


def _normalize(v: List[float]) -> List[float]:
    n = _l2_norm(v)
    return [x / n for x in v]


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def _tokens(s: str) -> List[str]:
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return [t for t in s.split() if t]


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


def _load_chunks_from_store() -> None:
    global CHUNK_BY_ID
    try:
        p = Path(_DOC_MAP_PATH)
        if not p.exists():
            INDEX.clear()
            CHUNK_BY_ID = {}
            return
        m = json.load(open(p, encoding="utf-8"))
        stored = m.get("chunks", {})
        CHUNK_BY_ID = {cid: DocChunk(**payload) for cid, payload in stored.items()}
        INDEX.clear()
        INDEX.extend(CHUNK_BY_ID.values())
    except Exception:
        INDEX.clear()
        CHUNK_BY_ID = {}


def _save_chunks_to_store(chunks: List[DocChunk]) -> None:
    """Persist chunks into the shared doc_map.json under 'chunks'."""
    try:
        p = Path(_DOC_MAP_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        m = {}
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                try:
                    m = json.load(f)
                except Exception:
                    m = {}
        m.setdefault("chunks", {})
        for c in chunks:
            m["chunks"][c.id] = c.dict()
        with p.open("w", encoding="utf-8") as f:
            json.dump(m, f)
    except Exception:
        pass


@app.on_event("startup")
def _warm_start() -> None:
    try:
        init_index(dim=get_index_dim())
    except Exception:
        pass
    _load_chunks_from_store()


def add_chunks(chunks: List[DocChunk]) -> int:
    """Embed and add chunks to both lexical and dense indexes."""
    if not chunks:
        return 0
    texts = [c.text for c in chunks]
    vecs = embed_texts_auto(texts, dim=get_index_dim(), batch_size=16)
    count = 0
    try:
        upsert_vectors([c.id for c in chunks], vecs)
    except Exception:
        pass
    for c, v in zip(chunks, vecs):
        INDEX.append(c)
        nv = _normalize([float(x) for x in v])
        DENSE_VECTORS.append(nv)
        VEC_BY_CHUNK_ID[c.id] = nv
        count += 1
    try:
        _save_chunks_to_store(chunks)
    except Exception:
        pass
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
    query_tokens = _tokens(req.query)
    candidates: List[DocChunk] = [
        c for c in INDEX if _allow_chunk(getattr(req, "filters", None), c)
    ]
    if not query_tokens or not candidates:
        return RetrieveResponse(hits=[])

    doc_terms: List[List[str]] = [_tokens(c.text) for c in candidates]
    doc_lens: List[int] = [max(1, len(ts)) for ts in doc_terms]
    avgdl = sum(doc_lens) / len(doc_lens)
    N = len(candidates)

    k1, b = 1.5, 0.75

    unique_q = set(query_tokens)
    df = {t: sum(1 for terms in doc_terms if t in set(terms)) for t in unique_q}
    idf = {
        t: math.log((N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1.0)
        for t in unique_q
    }

    q_vecs = embed_texts_auto([req.query], dim=get_index_dim(), batch_size=1)
    qv = _normalize([float(x) for x in q_vecs[0]]) if q_vecs else []

    faiss_scores: Dict[str, float] = {}
    if getattr(req, "hybrid", False) and qv:
        try:
            for cid, score in faiss_search(
                qv, top_k=(getattr(req, "dense_candidates", 0) or 50)
            ):
                faiss_scores[cid] = float(score)
        except Exception:
            faiss_scores = {}

    results: List[RetrieveResult] = []
    for i, chunk in enumerate(candidates):
        terms = doc_terms[i]
        dl = doc_lens[i]
        bm25_score = 0.0
        for t in unique_q:
            tf = terms.count(t)
            if tf == 0:
                continue
            numerator = tf * (k1 + 1.0)
            denominator = tf + k1 * (1.0 - b + b * (dl / avgdl))
            bm25_score += idf[t] * (numerator / denominator)
        cv = VEC_BY_CHUNK_ID.get(chunk.id)
        dense_score = _cosine(qv, cv) if cv else faiss_scores.get(chunk.id, 0.0)
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
        "bm25_scores": {h.chunk.id: h.bm25 for h in results},
    }
    return RetrieveResponse(hits=top_hits, diagnostics=diag)


@app.get("/debug/storage")
def debug_storage():
    idx = os.getenv("FAISS_INDEX_PATH", "/app/data/faiss.index")
    doc = os.getenv("DOC_MAP_PATH", "/app/data/doc_map.json")
    base = "/app/data"
    return {
        "cwd": os.getcwd(),
        "faiss_index_path": idx,
        "doc_map_path": doc,
        "exists": {
            "faiss.index": os.path.exists(idx),
            "doc_map.json": os.path.exists(doc),
        },
        "data_dir_listing": sorted(
            [os.path.basename(p) for p in glob.glob(base + "/*")]
        ),
    }
