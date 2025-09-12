"""Entry point for the retrieval microservice.

This service maintains an in‑memory index of document chunks and
implements a very simple retrieval algorithm based on word overlap.
In a full system you would employ BM25, dense vector search and
re‑ranking. The index API allows external components to register
chunks for later searching.
"""

from __future__ import annotations

from fastapi import FastAPI
from typing import List

from shared.models import (
    DocChunk,
    RetrieveRequest,
    RetrieveResponse,
    RetrieveResult,
)


app = FastAPI(title="Retrieval Service", version="0.1.0")

# In‑memory list of indexed chunks
INDEX: List[DocChunk] = []


@app.post("/index")
async def index_chunks(chunks: List[dict]) -> dict:
    """Add a list of chunks to the in‑memory index.

    This endpoint accepts raw dictionaries to avoid tight coupling to
    Pydantic models across services. Each item is parsed into a
    DocChunk instance and appended to the index.
    """
    count = 0
    for c in chunks:
        chunk = DocChunk.parse_obj(c)
        INDEX.append(chunk)
        count += 1
    return {"status": "ok", "indexed": count}


@app.post("/search", response_model=RetrieveResponse)
async def search(req: RetrieveRequest) -> RetrieveResponse:
    """Return a ranked list of document chunks matching the query.

    The current implementation scores each chunk by the proportion of
    query words that appear in the chunk. BM25 and dense retrieval
    methods can be added later by extending this function or by
    delegating to the appropriate backend.
    """
    query_terms = set(req.query.lower().split())
    results: List[RetrieveResult] = []
    for chunk in INDEX:
        text_terms = set(chunk.text.lower().split())
        if not query_terms:
            score = 0.0
        else:
            overlap = query_terms.intersection(text_terms)
            score = len(overlap) / len(query_terms)
        results.append(
            RetrieveResult(
                chunk=chunk,
                score=score,
                bm25=score,
                dense=score,
            )
        )
    # Sort in descending order of score
    results.sort(key=lambda r: r.score, reverse=True)
    top_hits = results[: req.top_k]
    return RetrieveResponse(hits=top_hits)