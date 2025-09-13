"""Embeddings microservice.

Exposes /embed to convert a list of DocChunk texts into dense vectors.
Behavior:
- Delegates to services.embeddings.app.embeddings.embed_texts (deterministic).
- Returns vectors and the configured embedding model name in EmbedResponse.

This service keeps no state and performs no storage; clients are expected
to persist vectors elsewhere if needed.
"""

from __future__ import annotations

from typing import Any, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from shared.settings import Settings
from shared.tracing import install_fastapi_tracing, span

from .embeddings import embed_texts

settings = Settings()  # ensure this exists in the file


class _FlexibleEmbedRequest(BaseModel):
    texts: Optional[List[str]] = None
    chunks: Optional[List[Any]] = None


app = FastAPI(title="Embeddings Service", version="0.1.0")
install_fastapi_tracing(app, service_name="embeddings")


@app.get("/")
def _root():
    return {"status": "ok", "service": "embeddings"}


@app.get("/health")
def _health():
    return {"status": "ok"}


@app.post("/embed")
async def embed(request: _FlexibleEmbedRequest) -> dict:
    """Generate embeddings for a list of texts. Accepts {'texts': [...] } or {'chunks': [...] }."""
    if request.texts:
        texts = request.texts
    elif request.chunks:
        try:
            texts = [
                (
                    getattr(c, "text", None)
                    or (c.get("text") if isinstance(c, dict) else "")
                )  # type: ignore
                for c in request.chunks
            ]
        except Exception:
            texts = []
    else:
        texts = []

    with span("embeddings.embed", num_texts=len(texts), model=settings.embedding_model):
        vectors = embed_texts(texts)

    # Return both keys for compatibility with different callers
    return {
        "vectors": vectors,
        "embeddings": vectors,
        "model": settings.embedding_model,
        "dim": (len(vectors[0]) if vectors else 0),
    }
