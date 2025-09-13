"""Embedding helpers (Gemini-only).

This module returns dense vectors for input texts with the following strategy:
- If Settings.offline_mode is True, produce stable, deterministic vectors
  derived from SHA-256 (useful for reproducible tests).
- Else if a Gemini API key is configured, call Gemini's embed_content API.
  A simple batching path is attempted first; it gracefully falls back to
  per-item calls if batching is unsupported or fails.
- Otherwise, generate random vectors with NumPy as a last resort.

Details:
- Default output dimensionality follows Settings.embedding_dim (fallback: 128).
- The code tolerates varied Gemini responses (dict/list/object forms) and
  normalizes them to List[List[float]].
- Used by both the embeddings service and retrieval for query embeddings.

Environment:
- GEMINI_API_KEY
- GEMINI_EMBEDDING_MODEL (default: gemini-embedding-001)
"""

from __future__ import annotations

import hashlib
import os
from typing import List, Optional

import numpy as np

from shared.settings import Settings

settings = Settings()

_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
_GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

genai = None
if _GEMINI_API_KEY and not settings.offline_mode:
    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=_GEMINI_API_KEY)
    except Exception:
        genai = None


def _deterministic_embed(text: str, dim: int = 128) -> List[float]:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    buf = (h * ((dim // len(h)) + 1))[:dim]
    return [b / 255.0 for b in buf]


def embed_texts(
    texts: List[str], dim: int = 128, batch_size: Optional[int] = None
) -> List[List[float]]:
    """Return embeddings for texts using Gemini if available; else random.

    If possible, uses simple batching to reduce API calls. Falls back to
    per‑item requests if batch call is unsupported or fails.
    """
    # Use Settings-derived defaults
    try:
        s = Settings()
        dim = s.embedding_dim or dim
        if s.offline_mode:
            return [_deterministic_embed(t, dim) for t in texts]
    except Exception:
        pass

    if genai is None:
        return [np.random.rand(dim).tolist() for _ in texts]

    vectors: List[List[float]] = []
    bsz = batch_size or max(16, len(texts))
    # Try batch path first
    try:
        if hasattr(genai, "embed_content") and bsz > 1:
            for i in range(0, len(texts), bsz):
                chunk = texts[i : i + bsz]
                try:
                    resp = genai.embed_content(model=_GEMINI_EMBED_MODEL, content=chunk)
                    # Common shape: { embeddings: [ { values: [...] }, ...] }
                    data = None
                    if isinstance(resp, dict):
                        data = resp.get("embeddings") or resp.get("data")
                    if data and isinstance(data, list):
                        for item in data:
                            vec = (
                                item.get("values")
                                or item.get("embedding")
                                or item.get("vector")
                            )
                            if (
                                not vec
                                and isinstance(item, dict)
                                and "embedding" in item
                            ):
                                emb = item.get("embedding") or {}
                                vec = emb.get("values") or emb.get("embedding")
                            vectors.append(
                                [
                                    float(x)
                                    for x in (vec or np.random.rand(dim).tolist())
                                ]
                            )
                    else:
                        # Unknown shape; fall back to per‑item
                        raise ValueError("Unexpected batch embedding response shape")
                except Exception:
                    # Batch failed; fall back to per‑item for this chunk
                    for t in chunk:
                        try:
                            single = genai.embed_content(
                                model=_GEMINI_EMBED_MODEL, content=t
                            )
                            vec = None
                            if isinstance(single, dict):
                                vec = (
                                    single.get("values")
                                    or (single.get("embedding") or {}).get("values")
                                    or (single.get("embedding") or {}).get("embedding")
                                )
                            else:
                                emb = getattr(single, "embedding", None)
                                vec = (
                                    getattr(emb, "values", None)
                                    if emb is not None
                                    else None
                                )
                            vectors.append(
                                [
                                    float(x)
                                    for x in (vec or np.random.rand(dim).tolist())
                                ]
                            )
                        except Exception:
                            vectors.append(np.random.rand(dim).tolist())
        else:
            raise ValueError("Batch path not supported; using single requests")
    except Exception:
        # Per‑item path
        vectors = []
        for t in texts:
            try:
                single = genai.embed_content(model=_GEMINI_EMBED_MODEL, content=t)
                vec = None
                if isinstance(single, dict):
                    vec = (
                        single.get("values")
                        or (single.get("embedding") or {}).get("values")
                        or (single.get("embedding") or {}).get("embedding")
                    )
                else:
                    emb = getattr(single, "embedding", None)
                    vec = getattr(emb, "values", None) if emb is not None else None
                vectors.append(
                    [float(x) for x in (vec or np.random.rand(dim).tolist())]
                )
            except Exception:
                vectors.append(np.random.rand(dim).tolist())
    return vectors
