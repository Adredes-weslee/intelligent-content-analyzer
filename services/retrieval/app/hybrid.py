"""Hybrid retrieval functions.

This module provides wrappers around the basic retrieval functionality
defined in `main.py`. In a production system you could combine BM25
scores with dense vector similarities using reciprocal rank fusion (RRF)
or similar techniques. Here we simply call the search endpoint and
return the result unchanged.
"""

from __future__ import annotations

from typing import List

from shared.models import RetrieveRequest, RetrieveResult, RetrieveResponse
from .main import search as simple_search


async def hybrid_search(request: RetrieveRequest) -> RetrieveResponse:
    """Perform a hybrid search combining keyword and dense retrieval.

    For now this function just calls the simple overlap search. It is
    defined separately so that more sophisticated logic can be added
    later without changing the API contracts.
    """
    return await simple_search(request)