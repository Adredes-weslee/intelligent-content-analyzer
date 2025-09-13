"""Document summarisation router for the API gateway with caching.

Uses the LLM summarizer to produce a concise summary and key insights
for a document, grounded on its indexed chunks. Results are cached per
``doc_id`` for a configurable TTL to avoid repeated LLM calls. Falls
back gracefully when no chunks are available.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from services.llm_generate.app.main import summarize_document as llm_summarize
from services.retrieval.app.main import INDEX  # type: ignore
from shared.cache import get_default_cache
from shared.models import SummarizeRequest
from shared.settings import Settings
from shared.tracing import span

router = APIRouter()
_cache = get_default_cache()


@router.get("/document_summary")
async def document_summary(doc_id: str, max_chunks: int | None = None) -> dict:
    """Summarize a document by doc_id using the LLM summarizer."""
    s = Settings()
    cache_key = f"summary:{doc_id}"
    if s.cache_enabled:
        cached = _cache.get(cache_key)
        if cached:
            return cached
    limit = max_chunks or s.summarizer_max_chunks
    with span("summary.collect_chunks", doc_id=doc_id, max_chunks=limit):
        chunks = [c for c in INDEX if getattr(c, "doc_id", None) == doc_id]
        if not chunks:
            raise HTTPException(status_code=404, detail="Document not found")
        chunks = chunks[:limit]

    with span("summary.llm"):
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
