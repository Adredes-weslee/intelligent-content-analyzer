"""Document summarisation router for the API gateway with caching.

Uses the LLM summarizer to produce a concise summary and key insights
for a document, grounded on its indexed chunks. Results are cached per
``doc_id`` for a configurable TTL to avoid repeated LLM calls.
"""

from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, HTTPException

from shared.cache import get_default_cache
from shared.models import SummarizeRequest
from shared.settings import Settings
from shared.tracing import span

router = APIRouter()
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
    from services.llm_generate.app.main import (
        summarize_document as llm_summarize,  # type: ignore
    )
    from services.retrieval.app.main import INDEX  # type: ignore


async def _get_chunks_http(doc_id: str, max_chunks: int | None) -> list[dict]:
    params = {"doc_id": doc_id}
    if max_chunks:
        params["max_chunks"] = max_chunks
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5, read=60)) as client:
        r = await client.get(f"{RETRIEVAL_URL}/chunks_by_doc", params=params)
        r.raise_for_status()
        return r.json().get("chunks", [])


async def _summarize_http(req: SummarizeRequest) -> dict:
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5, read=180, write=60)
    ) as client:
        try:
            r = await client.post(f"{LLM_GENERATE_URL}/summarize", json=req.dict())
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            # Surface upstream error body to caller
            detail = f"LLM summarize failed: {e.response.status_code} {e.response.text}"
            raise HTTPException(status_code=502, detail=detail)
        data = r.json()
        return {
            "summary": data.get("summary"),
            "key_points": data.get("key_points"),
            "citations": data.get("citations") or [],
            "diagnostics": data.get("diagnostics"),
        }


@router.get("/document_summary")
async def document_summary(doc_id: str, max_chunks: int | None = None) -> dict:
    """Summarize a document by doc_id using the LLM summarizer."""
    s = _settings
    cache_key = f"summary:{doc_id}"
    if s.cache_enabled:
        cached = _cache.get(cache_key)
        if cached:
            return cached

    limit = max_chunks or s.summarizer_max_chunks

    if _USE_HTTP:
        with span("summary.collect_chunks.http", doc_id=doc_id, max_chunks=limit):
            chunks = await _get_chunks_http(doc_id, limit)
            if not chunks:
                raise HTTPException(status_code=404, detail="Document not found")
        with span("summary.llm.http"):
            sreq = SummarizeRequest(doc_id=doc_id, chunks=chunks)  # type: ignore[arg-type]
            try:
                result = await _summarize_http(sreq)
            except HTTPException:
                raise
            except Exception as e:
                # Catch-all with minimal leak
                raise HTTPException(
                    status_code=502, detail=f"Summarize call failed: {e}"
                )
    else:
        with span("summary.collect_chunks.local", doc_id=doc_id, max_chunks=limit):
            chunks = [c for c in INDEX if getattr(c, "doc_id", None) == doc_id]
            if not chunks:
                raise HTTPException(status_code=404, detail="Document not found")
            chunks = chunks[:limit]
        with span("summary.llm.local"):
            sreq = SummarizeRequest(doc_id=doc_id, chunks=chunks)  # type: ignore[arg-type]
            sresp = await llm_summarize(sreq)
            result = {
                "summary": sresp.summary,
                "key_points": sresp.key_points,
                "citations": [c.dict() for c in sresp.citations],
                "diagnostics": sresp.diagnostics,
            }

    if s.cache_enabled:
        try:
            _cache.set(cache_key, result, ttl=s.summary_cache_ttl_seconds)
        except Exception:
            pass
    return result
