"""Confidence scoring utilities.

This module exposes a single helper, ``compute_confidence()``, that
produces a confidence score in ``[0.0, 1.0]`` for a generated answer.

How the score is computed (pure, no I/O):
- Retrieval component: mean of ``RetrieveResult.score`` across
    ``retrieval_hits``; if no hits are provided, this term is ``0.0``.
- Evaluation component: linear combination of values from ``eval_scores``
    for the keys ``"factuality"``, ``"relevance"``, ``"completeness"``, and optionally
    extended metrics ``"faithfulness"``, ``"context_relevance_ratio"``, and
    ``"answer_relevance_1_5"`` (normalized to 0–1 internally);
    any missing key is treated as ``0.0``.
- Optional judge component: if ``judge_scores`` is provided, the mean of
    the same three keys (missing keys ignored) is multiplied by the judge
    weight; if no valid judge values are present, this term is ``0.0``.

Weights and defaults:
- Callers may provide a ``weights`` dict with keys ``"retrieval"``,
    ``"factuality"``, ``"relevance"``, ``"completeness"``, optional
    ``"faithfulness"``, ``"context_relevance"``, ``"answer_relevance"``, and ``"judge"``.
- When ``weights`` is not provided, the following defaults are used:
    ``{"retrieval": 0.35, "factuality": 0.20, "relevance": 0.20, "completeness": 0.20, "judge": 0.05}``.
- Unrecognized weight keys are ignored. Missing keys default to ``0.0``.
- Weights are not re-normalized inside the function.

Assumptions and guarantees:
- Inputs for metrics are expected to be normalized to ``[0, 1]`` upstream.
- The final score is clamped to the inclusive range ``[0.0, 1.0]``.
- The function has no side effects and does not perform logging, I/O,
    or tracing.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from shared.models import RetrieveResult


def compute_confidence(
    retrieval_hits: List[RetrieveResult],
    eval_scores: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    judge_scores: Optional[Dict[str, float]] = None,
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
            "retrieval": 0.3,
            "factuality": 0.2,
            "relevance": 0.15,
            "completeness": 0.15,
            "faithfulness": 0.1,
            "context_relevance": 0.05,
            "answer_relevance": 0.05,
            "judge": 0.0,
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
        + weights.get("faithfulness", 0.0) * eval_scores.get("faithfulness", 0.0)
        + weights.get("context_relevance", 0.0)
        * eval_scores.get("context_relevance_ratio", 0.0)
        + weights.get("answer_relevance", 0.0)
        * (
            max(
                0.0,
                min(1.0, (eval_scores.get("answer_relevance_1_5", 0.0) - 1.0) / 4.0),
            )
        )
    )
    # Incorporate judge if available: mean of provided judge metrics
    if judge_scores:
        js_vals = [
            v
            for k, v in judge_scores.items()
            if k in ("factuality", "relevance", "completeness")
        ]
        if js_vals:
            confidence += weights.get("judge", 0.0) * (sum(js_vals) / len(js_vals))
    # Clamp to [0,1]
    return max(0.0, min(1.0, confidence))
