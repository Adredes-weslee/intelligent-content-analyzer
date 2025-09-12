"""Content chunking utilities.

Splitting documents into manageable chunks is critical for retrieval
systems that rely on vector embeddings. The chunk size should balance
between being large enough to capture context and small enough to fit
within model token limits. This module provides a simple word count
based splitter with a configurable maximum length.
"""

from __future__ import annotations

from typing import List


def chunk_text(text: str, max_tokens: int = 200) -> List[str]:
    """Split raw text into a list of chunks.

    The naive splitter operates on whitespace and counts tokens as
    whitespace‑separated words. It accumulates words until the token
    count exceeds `max_tokens` then emits a chunk. Overlap could be
    introduced later by sliding window if needed.

    Args:
        text: The full text to split.
        max_tokens: Rough limit on the number of words per chunk.

    Returns:
        A list of chunk strings.
    """
    words = text.split()
    chunks: List[str] = []
    buffer: List[str] = []
    for word in words:
        buffer.append(word)
        if len(buffer) >= max_tokens:
            chunks.append(" ".join(buffer))
            buffer = []
    if buffer:
        chunks.append(" ".join(buffer))
    return chunks