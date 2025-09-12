"""Confidence scoring utilities.

This module contains helper functions to compute a single confidence
score for a generated answer. The score is derived from a weighted
combination of retrieval scores and evaluation metrics. At present
weights are fixed, but they could be exposed via configuration in
future iterations.
"""

from __future__ import annotations

from typing import List, Dict

from shared.models import RetrieveResult


def compute_confidence(
    retrieval_hits: List[RetrieveResult],
    eval_scores: Dict[str, float],
    weights: Dict[str, float] | None = None,
) -> float:
    """Compute a confidence score between 0 and 1.

    Args:
        retrieval_hits: The search results used to answer the question.
        eval_scores: A dict with keys 'factuality', 'relevance',
            'completeness' representing evaluation metrics.
        weights: Optional weights for each component. Defaults to
            equal weighting.

    Returns:
        A float confidence score.
    """
    if not weights:
        weights = {
            "retrieval": 0.4,
            "factuality": 0.2,
            "relevance": 0.2,
            "completeness": 0.2,
        }
    # Compute average retrieval score
    if retrieval_hits:
        avg_retrieval = sum(hit.score for hit in retrieval_hits) / len(retrieval_hits)
    else:
        avg_retrieval = 0.0
    # Weighted sum
    confidence = (
        weights.get("retrieval", 0.0) * avg_retrieval
        + weights.get("factuality", 0.0) * eval_scores.get("factuality", 0.0)
        + weights.get("relevance", 0.0) * eval_scores.get("relevance", 0.0)
        + weights.get("completeness", 0.0) * eval_scores.get("completeness", 0.0)
    )
    # Clamp to [0,1]
    return max(0.0, min(1.0, confidence))