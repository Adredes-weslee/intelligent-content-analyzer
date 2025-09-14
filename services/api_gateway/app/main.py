"""API gateway for the Intelligent Content Analyzer.

This service exposes a unified REST interface to the outside world and
delegates work to the underlying microservices. It hides the
complexity of ingestion, retrieval, generation and evaluation behind
three simple endpoints.

Local/dev mode:
- The gateway composes synchronous calls to the Python modules directly
  rather than making network requests. This keeps the example small and
  avoids the need to start all services during testing.

Cloud/HTTP mode:
- In a real deployment the gateway sends HTTP requests to the other services.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shared.settings import Settings
from shared.tracing import install_fastapi_tracing

from .routers import qa, summary, upload
from .routers.debug import router as debug_router  # NEW

# Optional in-proc retrieval imports (used only in local/dev mode)
try:
    from services.retrieval.app.faiss_store import (
        _DOC_MAP_PATH as _RETRIEVAL_DOC_MAP,
    )
    from services.retrieval.app.faiss_store import get_index_dim as _retrieval_get_dim
    from services.retrieval.app.faiss_store import (
        get_index_dim as _retrievel_get_dim,  # noqa: F401 (spelling kept stable)
    )
    from services.retrieval.app.faiss_store import init_index as _retrieval_init_index
    from services.retrieval.app.main import INDEX as RETRIEVAL_INDEX
    from services.retrieval.app.main import _load_chunks_from_store as _retrieval_load
except Exception:
    _retrieval_load = None
    RETRIEVAL_INDEX = None
    _retrieval_init_index = None
    _retrieval_get_dim = None
    _RETRIEVAL_DOC_MAP = None

app = FastAPI(title="API Gateway", version="0.1.0")
install_fastapi_tracing(app, service_name="api-gateway")


# Health/root endpoints
@app.get("/")
def _root():
    return {"status": "ok", "service": "api-gateway"}


@app.get("/health")
def _health():
    return {"status": "ok"}


# -------- CORS (allow Streamlit origin) --------
s = Settings()
origin_env = (
    os.getenv("STREAMLIT_APP_ORIGIN") or getattr(s, "streamlit_app_origin", "") or ""
)
# Support comma-separated list if multiple origins are provided
origins = [o.strip().rstrip("/") for o in origin_env.split(",") if o.strip()]

# Fallback to your previous explicit defaults if nothing configured
if not origins:
    origins = [
        "https://adredes-weslee-intelligent-content-analyzer-uiapp-stwg9a.streamlit.app",
        "http://localhost:8501",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ----------------------------------------------


# Routers
app.include_router(upload.router, prefix="", tags=["upload"])
app.include_router(qa.router, prefix="", tags=["qa"])
app.include_router(summary.router, prefix="", tags=["summary"])
app.include_router(debug_router)  # NEW


# Determine mode: HTTP microservices vs local in-proc
USE_HTTP = any(
    [
        (os.getenv("INGEST_URL") or s.ingest_url),
        (os.getenv("RETRIEVAL_URL") or s.retrieval_url),
        (os.getenv("LLM_GENERATE_URL") or s.llm_generate_url),
        (os.getenv("EVALUATION_URL") or s.evaluation_url),
    ]
)


@app.on_event("startup")
def _bootstrap_retrieval_inproc() -> None:
    # Skip local bootstrap when using HTTP upstream services (cloud mode)
    if USE_HTTP:
        return
    if _retrieval_load and _retrieval_init_index and _retrieval_get_dim:
        try:
            _retrieval_init_index(dim=_retrieval_get_dim())
        except Exception:
            pass
        try:
            _retrieval_load()
        except Exception:
            pass


@app.get("/_retrieval_status")
def _retrieval_status():
    # In cloud mode, query the retrieval service over HTTP.
    if USE_HTTP:
        base = (os.getenv("RETRIEVAL_URL") or s.retrieval_url or "").rstrip("/")
        if not base:
            return {"error": "RETRIEVAL_URL not configured"}
        try:
            r = requests.get(f"{base}/status", timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    # Local/dev mode: inspect in-proc retrieval state/files.
    abs_path = str(Path(_RETRIEVAL_DOC_MAP).resolve()) if _RETRIEVAL_DOC_MAP else None
    chunk_count = None
    if _RETRIEVAL_DOC_MAP and Path(_RETRIEVAL_DOC_MAP).exists():
        try:
            m = json.load(open(_RETRIEVAL_DOC_MAP, encoding="utf-8"))
            chunk_count = len(m.get("chunks", {}))
        except Exception:
            chunk_count = -1
    try:
        import services.retrieval.app.main as _ret

        indexed_len = len(_ret.INDEX)
    except Exception:
        indexed_len = len(RETRIEVAL_INDEX) if RETRIEVAL_INDEX is not None else None
    return {
        "indexed": indexed_len,
        "doc_map_path": _RETRIEVAL_DOC_MAP,
        "doc_map_path_abs": abs_path,
        "doc_map_exists": (
            Path(_RETRIEVAL_DOC_MAP).exists() if _RETRIEVAL_DOC_MAP else None
        ),
        "doc_map_chunks": chunk_count,
    }


@app.on_event("startup")
async def _log_upstreams() -> None:
    upstreams = {
        "INGEST_URL": (os.getenv("INGEST_URL") or s.ingest_url or "").rstrip("/"),
        "RETRIEVAL_URL": (os.getenv("RETRIEVAL_URL") or s.retrieval_url or "").rstrip(
            "/"
        ),
        "EMBEDDINGS_URL": (
            os.getenv("EMBEDDINGS_URL") or s.embeddings_url or ""
        ).rstrip("/"),
        "LLM_GENERATE_URL": (
            os.getenv("LLM_GENERATE_URL") or s.llm_generate_url or ""
        ).rstrip("/"),
        "EVALUATION_URL": (
            os.getenv("EVALUATION_URL") or s.evaluation_url or ""
        ).rstrip("/"),
    }
    print(f"[api-gateway] Upstream services: {upstreams}")
