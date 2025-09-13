"""Cross-encoder and heuristic rerankers (cross-encoder default).

Provides:
- Cross-encoder reranker (sentence-transformers), using a MS MARCO model
  to score (query, chunk) pairs with fine-grained relevance.
- Heuristic fallback that boosts token overlap with the query.

Backend is controlled by Settings.reranker_backend and model via
Settings.reranker_model. Returned scores are normalized to [0,1] when possible.
"""

from __future__ import annotations

import math
from typing import List, Optional

from shared.models import RetrieveResult
from shared.settings import Settings

_CE = None
_CE_NAME: Optional[str] = None


def _load_cross_encoder(model_name: str) -> None:
    global _CE, _CE_NAME
    if _CE and _CE_NAME == model_name:
        return
    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        _CE = CrossEncoder(model_name)
        _CE_NAME = model_name
    except Exception:
        _CE = None
        _CE_NAME = None


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return max(0.0, min(1.0, x))


def _heuristic_rerank(
    results: List[RetrieveResult], query: str
) -> List[RetrieveResult]:
    if not query:
        return results
    q_tokens = [t for t in query.lower().split() if t]
    if not q_tokens:
        return results
    q_set = set(q_tokens)
    rescored: List[RetrieveResult] = []
    for r in results:
        t_set = set(r.chunk.text.lower().split())
        overlap = len(q_set.intersection(t_set))
        length_norm = max(1, len(t_set))
        fine_score = overlap / length_norm
        new_score = 0.5 * r.score + 0.5 * fine_score
        rescored.append(
            RetrieveResult(chunk=r.chunk, score=new_score, bm25=r.bm25, dense=r.dense)
        )
    rescored.sort(key=lambda x: x.score, reverse=True)
    return rescored


def rerank(
    results: List[RetrieveResult], model: str | None = None, query: str | None = None
) -> List[RetrieveResult]:
    """Rerank results using cross-encoder if selected/available, else heuristic."""
    s = Settings()
    backend = (s.reranker_backend or "heuristic").lower()
    ce_name = model or s.reranker_model
    if backend == "cross-encoder" and query:
        _load_cross_encoder(ce_name)
        if _CE is not None:
            pairs = [(query or "", r.chunk.text) for r in results]
            try:
                raw_scores = _CE.predict(pairs)  # type: ignore
                norm_scores = []
                for v in raw_scores:
                    vf = float(v)
                    norm_scores.append(vf if 0.0 <= vf <= 1.0 else _sigmoid(vf))
                rescored: List[RetrieveResult] = []
                for r, score in zip(results, norm_scores):
                    rescored.append(
                        RetrieveResult(
                            chunk=r.chunk,
                            score=float(score),
                            bm25=r.bm25,
                            dense=r.dense,
                        )
                    )
                rescored.sort(key=lambda x: x.score, reverse=True)
                return rescored
            except Exception:
                pass
    return _heuristic_rerank(results, query or "")
