# services/api_gateway/app/routers/debug.py

from __future__ import annotations

import os
from typing import Dict

import httpx
from fastapi import APIRouter

from shared.settings import Settings

router = APIRouter(tags=["debug"])


async def _probe(url: str) -> Dict[str, str]:
    if not url:
        return {"status": "unconfigured"}
    try:
        u = url.rstrip("/") + "/health"
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5, read=5)) as c:
            r = await c.get(u)
            return {"status": str(r.status_code), "body": (r.text[:200] or "")}
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}


@router.get("/debug/upstreams", include_in_schema=False)
async def debug_upstreams():
    s = Settings()
    # Prefer env vars if set; fall back to Settings fields
    ingest = (os.getenv("INGEST_URL") or s.ingest_url or "").rstrip("/")
    retrieval = (os.getenv("RETRIEVAL_URL") or s.retrieval_url or "").rstrip("/")
    embeddings = (os.getenv("EMBEDDINGS_URL") or s.embeddings_url or "").rstrip("/")
    llm = (os.getenv("LLM_GENERATE_URL") or s.llm_generate_url or "").rstrip("/")
    evaluation = (os.getenv("EVALUATION_URL") or s.evaluation_url or "").rstrip("/")

    results = {
        "ingest": {"url": ingest, **(await _probe(ingest))},
        "retrieval": {"url": retrieval, **(await _probe(retrieval))},
        "embeddings": {"url": embeddings, **(await _probe(embeddings))},
        "llm_generate": {"url": llm, **(await _probe(llm))},
        "evaluation": {"url": evaluation, **(await _probe(evaluation))},
    }
    return results
