"""Entry point for the embeddings microservice.

This service exposes a simple endpoint for converting document chunks
into dense vectors. In a production system you would integrate with a
state of the art embedding model such as OpenAI's `text-embedding-ada-002`
or Hugging Face's sentence transformers. Here we generate random
vectors to demonstrate the API contract.
"""

from __future__ import annotations

from fastapi import FastAPI
import numpy as np

from shared.models import EmbedRequest, EmbedResponse
from shared.settings import Settings


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
    dim = 128  # fixed embedding dimensionality for demonstration
    vectors = [np.random.rand(dim).tolist() for _ in request.chunks]
    return EmbedResponse(vectors=vectors, model=settings.embedding_model)