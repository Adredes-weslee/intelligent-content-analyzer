"""
Document summarisation router for the API gateway with caching.

Uses the LLM summarizer to produce a concise summary and key insights
for a document, grounded on its indexed chunks. Results are cached per
``doc_id`` for a configurable TTL to avoid repeated LLM calls.
"""

from __future__ import annotations

import os
from typing import Any, List

import httpx
from fastapi import APIRouter, HTTPException, Query

from shared.cache import get_default_cache
from shared.models import DocChunk, SummarizeRequest, SummarizeResponse
from shared.settings import Settings
from shared.tracing import span

router = APIRouter(tags=["summary"])
_cache = get_default_cache()
_settings = Settings()

RETRIEVAL_URL = (os.getenv("RETRIEVAL_URL") or _settings.retrieval_url or "").rstrip(
    "/"
)
LLM_GENERATE_URL = (
    os.getenv("LLM_GENERATE_URL") or _settings.llm_generate_url or ""
).rstrip("/")
_USE_HTTP = bool(RETRIEVAL_URL and LLM_GENERATE_URL)

# Local-only imports when running in single-process mode
if not _USE_HTTP:
    from services.llm_generate.app.main import (  # type: ignore
        summarize_document as llm_summarize,
    )
    from services.retrieval.app.main import INDEX  # type: ignore


async def _fetch_chunks_http(doc_id: str, max_chunks: int) -> list[dict]:
    """Fetch chunks for a document from the retrieval service with strict error mapping."""
    params = {"doc_id": doc_id, "max_chunks": max_chunks}
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10, read=60, write=60)
        ) as client:
            r = await client.get(f"{RETRIEVAL_URL}/chunks_by_doc", params=params)
            r.raise_for_status()
            payload = r.json()
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        body = (e.response.text or "")[:500]
        if status == 404:
            raise HTTPException(404, "Document not found")
        if status in (400, 422):
            raise HTTPException(400, f"Bad request to retrieval: {body}")
        raise HTTPException(502, f"Retrieval failed: {status} {body}")
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Retrieval failed: {e}")

    raw = payload.get("chunks") if isinstance(payload, dict) else payload
    if not isinstance(raw, list) or not raw:
        raise HTTPException(404, f"No chunks found for doc_id={doc_id}")
    return raw


async def _summarize_http(req: SummarizeRequest) -> SummarizeResponse:
    """Call the LLM generation service to summarize, surfacing upstream error details."""
    if not LLM_GENERATE_URL:
        raise HTTPException(500, "LLM_GENERATE_URL not configured")
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10, read=180, write=180)
        ) as client:
            r = await client.post(
                f"{LLM_GENERATE_URL}/summarize", json=req.model_dump()
            )
            r.raise_for_status()
            return SummarizeResponse.model_validate(r.json())
    except httpx.HTTPStatusError as e:
        body = (e.response.text or "")[:500]
        raise HTTPException(
            502, f"{LLM_GENERATE_URL}/summarize -> {e.response.status_code}: {body}"
        )
    except httpx.HTTPError as e:
        raise HTTPException(502, f"{LLM_GENERATE_URL}/summarize -> {e}")


@router.get("/document_summary", response_model=SummarizeResponse)
async def document_summary(
    doc_id: str = Query(..., min_length=1),
    max_chunks: int | None = None,
) -> Any:
    """
    Summarize a document by doc_id using the LLM summarizer.

    Flow (HTTP mode):
      - GET chunks from retrieval
      - Cap to Settings.summarizer_max_chunks
      - POST SummarizeRequest(chunks=...) to llm-generate
    """
    s = _settings
    # Cache key is per doc; if you need per-max_chunks caching later, include max_chunks in the key.
    cache_key = f"summary:{doc_id}"

    if s.cache_enabled:
        cached = _cache.get(cache_key)
        if cached:
            return cached

    limit = int(max_chunks or s.summarizer_max_chunks or 12)

    if _USE_HTTP:
        with span("summary.collect_chunks.http", doc_id=doc_id, max_chunks=limit):
            raw_chunks = await _fetch_chunks_http(doc_id, limit)
            # Validate + cap to N as DocChunk (guards against schema drift)
            try:
                chunks: List[DocChunk] = [
                    DocChunk.model_validate(c) for c in raw_chunks[:limit]
                ]
            except Exception as e:
                raise HTTPException(500, f"chunk schema error: {e}")

        with span("summary.llm.http"):
            sreq = SummarizeRequest(doc_id=doc_id, chunks=chunks)
            sresp = await _summarize_http(sreq)
            result = {
                "summary": sresp.summary,
                "key_points": sresp.key_points,
                "citations": [c.model_dump() for c in (sresp.citations or [])],
                "diagnostics": sresp.diagnostics,
            }
    else:
        # Single-process mode (local dev): use in-memory retrieval + local LLM function
        with span("summary.collect_chunks.local", doc_id=doc_id, max_chunks=limit):
            local_chunks = [c for c in INDEX if getattr(c, "doc_id", None) == doc_id]
            if not local_chunks:
                raise HTTPException(status_code=404, detail="Document not found")
            local_chunks = local_chunks[:limit]

        with span("summary.llm.local"):
            sreq = SummarizeRequest(doc_id=doc_id, chunks=local_chunks)  # type: ignore[arg-type]
            sresp = await llm_summarize(sreq)
            result = {
                "summary": sresp.summary,
                "key_points": sresp.key_points,
                "citations": [c.dict() for c in (sresp.citations or [])],
                "diagnostics": sresp.diagnostics,
            }

    if s.cache_enabled:
        try:
            _cache.set(cache_key, result, ttl=s.summary_cache_ttl_seconds)
        except Exception:
            # Non-fatal if cache backend is unavailable
            pass

    return result
