"""Embedding helpers (Gemini-only).

This module returns dense vectors for input texts with the following strategy:
- If Settings.offline_mode is True, produce stable, deterministic vectors
  derived from SHA-256 (useful for reproducible tests).
- Else if a Gemini API key is configured, call Gemini's embed_content API.
  A simple batching path is attempted first; it gracefully falls back to
  per-item calls if batching is unsupported or fails.
- Otherwise, generate deterministic vectors from SHA-256 (not random), so
  repeated calls for the same text are identical.

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
from typing import List, Optional

from shared.settings import Settings

settings = Settings()

_GEMINI_API_KEY = settings.gemini_api_key
_GEMINI_EMBED_MODEL = settings.embedding_model

genai = None
if _GEMINI_API_KEY and not settings.offline_mode:
    try:
        # For the newer Google GenAI SDK:
        #   pip install google-genai
        #   from google import genai
        from google import genai  # type: ignore

        genai.configure(api_key=_GEMINI_API_KEY)
    except Exception:
        genai = None


def _deterministic_embed(text: str, dim: int = 768) -> List[float]:
    """Deterministic byte-based embedding from SHA-256, length = dim."""
    h = hashlib.sha256((text or "").encode("utf-8")).digest()
    buf = (h * ((dim // len(h)) + 1))[:dim]
    # Values in [0, 1]; stable across runs
    return [b / 255.0 for b in buf]


def embed_texts(
    texts: List[str], batch_size: int = 16, dim: Optional[int] = None
) -> List[List[float]]:
    """
    Return embeddings for texts.
    - If Settings.offline_mode: deterministic vectors with requested dim (or settings.embedding_dim).
    - If GEMINI available: request output_dimensionality=dim and use batch API when available.
      On any failure, fall back to deterministic embeddings (not random).
    - Else: deterministic embeddings.
    """
    global genai
    s = settings
    target_dim = int(dim or s.embedding_dim or 768)

    # Strictly deterministic in offline mode
    if s.offline_mode:
        return [_deterministic_embed(t, target_dim) for t in texts]

    if _GEMINI_API_KEY:
        if genai is None:
            try:
                from google import genai  # type: ignore

                genai.configure(api_key=_GEMINI_API_KEY)
            except Exception:
                genai = None
        if genai is not None:
            out: List[List[float]] = []
            bsz = max(1, int(batch_size or 16))
            try:
                # Batch if supported
                if hasattr(genai, "batch_embed_contents") and bsz > 1:
                    for i in range(0, len(texts), bsz):
                        chunk = texts[i : i + bsz]
                        try:
                            reqs = [
                                {"content": t, "output_dimensionality": target_dim}
                                for t in chunk
                            ]
                            resp = genai.batch_embed_contents(
                                model=_GEMINI_EMBED_MODEL, requests=reqs
                            )  # type: ignore[attr-defined]
                            data = None
                            if isinstance(resp, dict):
                                data = resp.get("embeddings") or resp.get("data")
                            elif hasattr(resp, "embeddings"):
                                data = getattr(resp, "embeddings", None)
                            if data and isinstance(data, list):
                                for item in data:
                                    vec = (
                                        item.get("values")
                                        if isinstance(item, dict)
                                        else None
                                    ) or (
                                        item.get("embedding")
                                        if isinstance(item, dict)
                                        else None
                                    )
                                    if isinstance(vec, dict):
                                        vec = vec.get("values") or vec.get("embedding")
                                    if not vec:
                                        raise ValueError("No embedding in batch item")
                                    out.append([float(x) for x in vec])
                            else:
                                raise ValueError("Unexpected batch response shape")
                        except Exception:
                            # Deterministic per-item fallback
                            for t in chunk:
                                try:
                                    single = genai.embed_content(
                                        model=_GEMINI_EMBED_MODEL,
                                        content=t,
                                        output_dimensionality=target_dim,
                                    )
                                    vec = None
                                    if isinstance(single, dict):
                                        emb = single.get("embedding")
                                        if isinstance(emb, dict):
                                            vec = emb.get("values") or emb.get(
                                                "embedding"
                                            )
                                        else:
                                            vec = (
                                                emb
                                                or single.get("values")
                                                or single.get("vector")
                                            )
                                    else:
                                        emb = getattr(single, "embedding", None)
                                        vec = (
                                            (emb.get("values") or emb.get("embedding"))
                                            if isinstance(emb, dict)
                                            else emb
                                        )
                                    if not vec:
                                        raise ValueError("No embedding in response")
                                    out.append([float(x) for x in vec])
                                except Exception:
                                    out.append(_deterministic_embed(t, target_dim))
                    return out
                else:
                    # Single-call path
                    for t in texts:
                        try:
                            single = genai.embed_content(
                                model=_GEMINI_EMBED_MODEL,
                                content=t,
                                output_dimensionality=target_dim,
                            )
                            vec = None
                            if isinstance(single, dict):
                                emb = single.get("embedding")
                                if isinstance(emb, dict):
                                    vec = emb.get("values") or emb.get("embedding")
                                else:
                                    vec = (
                                        emb
                                        or single.get("values")
                                        or single.get("vector")
                                    )
                            else:
                                emb = getattr(single, "embedding", None)
                                vec = (
                                    (emb.get("values") or emb.get("embedding"))
                                    if isinstance(emb, dict)
                                    else emb
                                )
                            if not vec:
                                raise ValueError("No embedding in response")
                            out.append([float(x) for x in vec])
                        except Exception:
                            out.append(_deterministic_embed(t, target_dim))
                    return out
            except Exception:
                # Fall through to deterministic below
                pass

    # Deterministic final fallback (no provider)
    return [_deterministic_embed(t, target_dim) for t in texts]
