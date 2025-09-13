"""Embeddings microservice.

Exposes /embed to convert a list of DocChunk texts into dense vectors.
Behavior:
- Delegates to services.embeddings.app.embeddings.embed_texts, which uses:
  - deterministic vectors in offline mode,
  - Gemini embeddings when configured,
  - or random vectors as a fallback.
- Returns vectors and the configured embedding model name in EmbedResponse.

This service keeps no state and performs no storage; clients are expected
to persist vectors elsewhere if needed.
"""

from __future__ import annotations

from fastapi import FastAPI

from shared.models import EmbedRequest, EmbedResponse
from shared.settings import Settings
from shared.tracing import log_event, span

from .embeddings import embed_texts

settings = Settings()
app = FastAPI(title="Embeddings Service", version="0.1.0")


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest) -> EmbedResponse:
    """Generate random embeddings for a list of document chunks.

    Args:
        request: The embedding request containing document chunks.

    Returns:
        An object containing a list of vectors and the model name.
    """
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
