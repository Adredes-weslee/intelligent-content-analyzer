"""Content chunking utilities.

Splitting documents into manageable chunks is critical for retrieval
systems that rely on vector embeddings. The chunk size should balance
between being large enough to capture context and small enough to fit
within model token limits. This module provides:
- A simple word count based splitter (`chunk_text`).
- A section-aware iterator (`iter_section_chunks`) that respects page
    boundaries ("=== Page N ==="), detects headings, and propagates table
    identifiers from markers like "[table id=...] ... [/table]".
"""

from __future__ import annotations

import re
from typing import Generator, List, Optional, Tuple


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


def _is_heading(line: str) -> Optional[str]:
    """Return normalized heading text if the line looks like a heading.

    Heuristics:
    - Markdown-style: leading # characters
    - Numbered headings: 1. or 1.2.3 style prefixes
    - Title-like lines: short lines with Title Case or ALL CAPS
    """
    s = line.strip()
    if not s:
        return None
    m = re.match(r"^(#{1,6})\s+(.+)$", s)
    if m:
        return m.group(2).strip()
    if re.match(r"^\d+(?:\.\d+)*\s+.+$", s):
        return s
    # Title-like: <= 80 chars, few punctuation, many caps
    if len(s) <= 80 and not s.endswith(":"):
        # Avoid page headers and table markers
        if s.startswith("=== Page ") or s.startswith("[table "):
            return None
        words = s.split()
        if 1 <= len(words) <= 12:
            uppers = sum(1 for ch in s if ch.isupper())
            letters = sum(1 for ch in s if ch.isalpha())
            if letters > 0 and uppers / max(1, letters) > 0.35:
                return s
    return None


def iter_section_chunks(
    text: str,
    max_tokens: int = 200,
    respect_pages: bool = True,
    respect_headings: bool = True,
) -> Generator[Tuple[str, Optional[int], Optional[str], Optional[str]], None, None]:
    """Yield section-aware chunks with page/section/table metadata.

    Yields tuples: (chunk_text, page_number, section_title, table_id)

    - Detects page headers emitted by readers: "=== Page N ===".
    - Detects table blocks delimited by markers: [table id=...] ... [/table]
    - Optionally updates current section title based on heading-like lines.
    """
    page = None
    section: Optional[str] = None
    token_buf: List[str] = []

    def flush_plain_buf() -> Optional[
        Tuple[str, Optional[int], Optional[str], Optional[str]]
    ]:
        nonlocal token_buf, page, section
        if not token_buf:
            return None
        chunk = " ".join(token_buf).strip()
        token_buf = []
        if chunk:
            return (chunk, page, section, None)
        return None

    # Handle table blocks separately to attach table_id
    i = 0
    lines = text.splitlines()
    while i < len(lines):
        line = lines[i]
        # Page header
        if respect_pages:
            m_page = re.match(r"^=== Page (\d+) ===$", line.strip())
            if m_page:
                # Flush any pending text before page switch
                flushed = flush_plain_buf()
                if flushed:
                    yield flushed
                page = int(m_page.group(1))
                i += 1
                continue
        # Table block
        if line.strip().startswith("[table id="):
            # Extract table_id
            m = re.match(r"^\[table id=([^\]]+)\]$", line.strip())
            table_id = m.group(1) if m else None
            # Accumulate table content until [/table]
            tbl_tokens: List[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() != "[/table]":
                tbl_tokens.extend(lines[i].strip().split())
                # Emit in chunks respecting max_tokens
                while len(tbl_tokens) >= max_tokens:
                    chunk = " ".join(tbl_tokens[:max_tokens])
                    yield (chunk, page, section, table_id)
                    tbl_tokens = tbl_tokens[max_tokens:]
                i += 1
            if tbl_tokens:
                yield (" ".join(tbl_tokens), page, section, table_id)
            # Skip closing [/table]
            if i < len(lines) and lines[i].strip() == "[/table]":
                i += 1
            continue

        # Heading detection
        if respect_headings:
            hd = _is_heading(line)
            if hd:
                # Flush previous paragraph buffer under old section
                flushed = flush_plain_buf()
                if flushed:
                    yield flushed
                section = hd
                i += 1
                continue

        # Normal text
        tokens = line.strip().split()
        for tok in tokens:
            token_buf.append(tok)
            if len(token_buf) >= max_tokens:
                flushed = flush_plain_buf()
                if flushed:
                    yield flushed
        # Blank line boundary: encourage flush
        if not tokens and token_buf:
            flushed = flush_plain_buf()
            if flushed:
                yield flushed
        i += 1

    # Flush remainder
    flushed = flush_plain_buf()
    if flushed:
        yield flushed
