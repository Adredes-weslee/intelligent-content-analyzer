"""Reranker utilities.

Re‑ranking is often used to refine the order of retrieved documents by
running a more expensive model over the top candidates. This module
contains a stub implementation that preserves the input order. A real
implementation could use a cross‑encoder or a large language model to
score (query, document) pairs.
"""

from __future__ import annotations

from typing import List

from shared.models import RetrieveResult


def rerank(results: List[RetrieveResult], model: str | None = None) -> List[RetrieveResult]:
    """Return the input list without modification.

    Args:
        results: A list of search results to rerank.
        model: The name of the reranker model (unused).

    Returns:
        The same list of results.
    """
    return results