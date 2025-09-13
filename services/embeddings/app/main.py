"""Embeddings microservice.

Exposes /embed to convert a list of DocChunk texts into dense vectors.
Behavior:
- Delegates to services.embeddings.app.embeddings.embed_texts (deterministic).
- Returns vectors and the configured embedding model name in EmbedResponse.

This service keeps no state and performs no storage; clients are expected
to persist vectors elsewhere if needed.
"""

from __future__ import annotations

from fastapi import FastAPI

from shared.models import EmbedRequest, EmbedResponse
from shared.settings import Settings
from shared.tracing import install_fastapi_tracing, log_event, span

from .embeddings import embed_texts

settings = Settings()
app = FastAPI(title="Embeddings Service", version="0.1.0")
install_fastapi_tracing(app, service_name="embeddings")


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest) -> EmbedResponse:
    """Generate embeddings for a list of document chunks."""
    texts = [c.text for c in request.chunks]
    with span("embeddings.embed", num_texts=len(texts), model=settings.embedding_model):
        vectors = embed_texts(texts)
    log_event(
        "Embedding",
        payload={
            "num_texts": len(texts),
            "model": settings.embedding_model,
            "dim": len(vectors[0]) if vectors else 0,
        },
        correlation_id=None,
    )
    return EmbedResponse(vectors=vectors, model=settings.embedding_model)
