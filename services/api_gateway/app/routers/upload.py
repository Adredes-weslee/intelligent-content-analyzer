"""Upload router for the API gateway.

Parses a minimal multipart/form-data request body to extract a single file
without requiring python-multipart. The file bytes are sent to the ingest
reader to obtain raw text, which is split into chunks and registered into
the in‑memory retrieval index.

Current behavior:
- parse_document: best-effort text extraction (simple decoding).
- chunk_text: word-based splitter with a fixed max length.
- Language detection (langdetect) is performed once per document and added
  to chunk metadata.
- Chunks are indexed immediately via services.retrieval.app.main.add_chunks.
- Tracing spans are emitted around parsing and indexing.

This router is stateless and returns a generated doc_id and chunk count.
"""

from __future__ import annotations

import datetime
import hashlib
import mimetypes
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from langdetect import detect  # type: ignore

from services.ingest.app.chunkers import chunk_text, iter_section_chunks
from services.ingest.app.readers import parse_document
from services.retrieval.app.faiss_store import (
    init_index as init_faiss_index,
)
from services.retrieval.app.faiss_store import (
    lookup_by_checksum,
    lookup_file,
    track_doc,
    update_file_entry,
)
from services.retrieval.app.faiss_store import (
    remove_chunks as faiss_remove_chunks,
)
from services.retrieval.app.main import add_chunks, remove_chunks_by_ids  # type: ignore
from shared.cache import bump_index_version
from shared.models import DocChunk, DocMetadata
from shared.settings import Settings
from shared.tracing import span

router = APIRouter()
_settings = Settings()
init_faiss_index()


@router.post(
    "/upload_document",
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {"file": {"type": "string", "format": "binary"}},
                        "required": ["file"],
                    }
                }
            }
        }
    },
)
async def upload_document(request: Request) -> JSONResponse:
    """Upload a document and return its generated ID.

    The API gateway accepts a multipart/form-data request containing a
    file field named ``file``. To avoid depending on the optional
    python-multipart package, this handler manually parses the raw
    request body to extract the uploaded file. The file contents are
    then processed by the same parsing and chunking logic used by the
    ingest service. Parsed chunks are added to the in-memory retrieval
    index for immediate availability in subsequent queries.

    Args:
        request: The incoming request containing a multipart body.

    Returns:
        A JSON response with a generated document ID and the number of
        chunks produced.
    """
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        raise HTTPException(
            status_code=400, detail="Content-Type must be multipart/form-data"
        )
    boundary_marker = "boundary="
    if boundary_marker not in content_type:
        raise HTTPException(status_code=400, detail="Missing multipart boundary")
    boundary = (
        content_type.split(boundary_marker, 1)[1].split(";", 1)[0].strip().strip('"')
    )
    body = await request.body()
    file_bytes = None
    filename = "uploaded"
    delimiter = b"--" + boundary.encode()
    parts = body.split(delimiter)
    for part in parts:
        if b"Content-Disposition" in part and b'name="file"' in part:
            header, _, data_section = part.partition(b"\r\n\r\n")
            data = data_section.rsplit(b"\r\n", 1)[0]
            header_str = header.decode(errors="ignore")
            import re as _re

            m_fn = _re.search(
                r'filename\*?=(?:"([^"]+)"|([^;\r\n]+))', header_str, _re.I
            )
            if m_fn:
                raw_fn = (m_fn.group(1) or m_fn.group(2)).strip()
                import os as _os

                filename = _os.path.basename(raw_fn)
            file_bytes = data
            break
    if not file_bytes:
        raise HTTPException(status_code=400, detail="No file part provided")
    checksum = hashlib.sha256(file_bytes).hexdigest()
    existing_doc = lookup_by_checksum(checksum)
    if existing_doc:
        return JSONResponse(
            {"doc_id": existing_doc, "num_chunks": 0, "idempotent": True}
        )
    with span("gateway.upload.parse_and_chunk", filename=filename):
        raw_text = parse_document(file_bytes, filename)
        file_entry = lookup_file(filename)
        doc_id = file_entry["doc_id"] if file_entry else str(uuid.uuid4())
        previous_chunk_ids = file_entry.get("chunks", []) if file_entry else []
        try:
            lang = detect(raw_text)[:2]
        except Exception:
            lang = None
        chunks: list[DocChunk] = []
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        created_at = datetime.datetime.utcnow().isoformat()
        if _settings.chunk_section_aware_enabled:
            iterator = iter_section_chunks(
                raw_text,
                max_tokens=_settings.chunk_max_tokens,
                respect_pages=_settings.chunk_respect_pages,
                respect_headings=_settings.chunk_respect_headings,
            )
            for idx, (text, page, section, table_id) in enumerate(iterator):
                meta = DocMetadata(
                    source=filename,
                    page=page,
                    section=section,
                    table_id=table_id,
                    lang=lang,
                    content_type=content_type,
                    checksum=checksum,
                    created_at=created_at,
                )
                chunks.append(
                    DocChunk(id=f"{doc_id}_{idx}", doc_id=doc_id, text=text, meta=meta)
                )
        else:
            text_chunks = chunk_text(raw_text, max_tokens=_settings.chunk_max_tokens)
            for idx, text in enumerate(text_chunks):
                meta = DocMetadata(
                    source=filename,
                    lang=lang,
                    content_type=content_type,
                    checksum=checksum,
                    created_at=created_at,
                )
                chunks.append(
                    DocChunk(id=f"{doc_id}_{idx}", doc_id=doc_id, text=text, meta=meta)
                )
    if previous_chunk_ids and _settings.enable_upsert:
        try:
            remove_chunks_by_ids(previous_chunk_ids)
        except Exception:
            pass
        try:
            faiss_remove_chunks(previous_chunk_ids)
        except Exception:
            pass
    add_chunks(chunks)
    chunk_ids = [c.id for c in chunks]
    try:
        if file_entry:
            update_file_entry(filename, checksum, chunk_ids)
        else:
            track_doc(filename, doc_id, checksum, chunk_ids)
    except Exception:
        pass
    try:
        bump_index_version()
    except Exception:
        pass
    return JSONResponse(
        {
            "doc_id": doc_id,
            "num_chunks": len(chunks),
            "upserted": bool(previous_chunk_ids),
        }
    )
