"""Placeholder judge module.

In a future implementation this module could wrap a large language model
to grade answers according to a rubric. For now it simply returns
fixed scores. Keeping the code here allows drop‑in replacement later.
"""

from __future__ import annotations

from typing import Tuple

from shared.models import EvaluateRequest


def llm_judge(request: EvaluateRequest) -> Tuple[float, float, float]:
    """Return dummy evaluation scores.

    Args:
        request: The evaluation request to judge.

    Returns:
        A tuple of three floats representing factuality, relevance and
        completeness.
    """
    # Always return mid‑range scores for demonstration purposes
    return 0.5, 0.5, 0.5