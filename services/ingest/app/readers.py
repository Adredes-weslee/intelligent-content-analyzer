"""Document readers for the ingest microservice.

This module defines helper functions to extract raw text from various file
formats. In the initial version we support plain text extraction by
decoding bytes. More sophisticated extraction (PDF, DOCX, images) can be
added here later. The goal is to provide a best‑effort textual
representation of the input for downstream processing.
"""

from __future__ import annotations

from typing import Optional


def parse_document(content: bytes, filename: str) -> str:
    """Parse a document into a string.

    For now we assume the document is UTF‑8 encoded text. In the future
    this function could dispatch based on file extension and use
    specialised libraries (e.g. pdfminer.six for PDF, python-docx for
    Word documents, pytesseract for scanned images).

    Args:
        content: The raw bytes of the file.
        filename: The name of the file; unused but could hint at format.

    Returns:
        A unicode string representation of the document.
    """
    # Attempt to decode as UTF‑8. Replace undecodable bytes.
    try:
        return content.decode("utf-8", errors="replace")
    except Exception:
        # Fallback: decode as latin-1 which never fails
        return content.decode("latin-1", errors="replace")