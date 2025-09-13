"""Confidence scoring utilities.

Blends retrieval signals, heuristic evaluation scores, and optional judge
scores into a single confidence value in [0,1].
"""

from __future__ import annotations

from typing import Dict, Optional


def _clip01(x: float) -> float:
    try:
        if x != x:
            return 0.0
        return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)
    except Exception:
        return 0.0


def _avg_clip01(d: Dict[str, float]) -> float:
    if not d:
        return 0.0
    vals = [_clip01(v) for v in d.values()]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def compute_confidence(
    *,
    retrieval_top: float,
    retrieval_mean: float,
    eval_scores: Dict[str, float],
    judge_scores: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Blend retrieval signals, evaluation metrics, and optional judge scores into [0,1].

    Parameters
    - retrieval_top: top hit score (normalized or will be clipped to [0,1])
    - retrieval_mean: mean score over the used top-k (normalized or clipped)
    - eval_scores: dict with keys like factuality, relevance, completeness, faithfulness
    - judge_scores: optional dict of judge-derived scores (e.g., entailment/faithfulness)
    - weights: optional override of weighting

    Returns
    - confidence in [0,1]
    """
    w = {
        "retrieval_top": 0.25,
        "retrieval_mean": 0.10,
        "factuality": 0.25,
        "relevance": 0.15,
        "completeness": 0.10,
        "faithfulness": 0.10,
        "judge": 0.01,
        "ctx_ratio": 0.09,
    }
    if weights:
        w.update(weights)

    rt = _clip01(retrieval_top)
    rm = _clip01(retrieval_mean)
    if rt < 0.02 and rm < 0.02:
        w["judge"] = 0.0

    fact = _clip01(eval_scores.get("factuality", 0.0))
    rel = _clip01(eval_scores.get("relevance", 0.0))
    comp = _clip01(eval_scores.get("completeness", 0.0))
    faith = _clip01(eval_scores.get("faithfulness", 0.0))
    ctx = _clip01(eval_scores.get("context_relevance_ratio", 0.0))

    judge_avg = _avg_clip01(judge_scores or {})

    score = (
        w["retrieval_top"] * rt
        + w["retrieval_mean"] * rm
        + w["factuality"] * fact
        + w["relevance"] * rel
        + w["completeness"] * comp
        + w["faithfulness"] * faith
        + w["judge"] * judge_avg
        + w["ctx_ratio"] * ctx
    )

    if ctx < 0.15 and rt < 0.05 and rm < 0.05:
        score = min(score, 0.35)

    return _clip01(score)
