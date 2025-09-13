"""Ingest microservice.

Accepts `multipart/form-data` uploads, parses files into normalized text and
emits chunks. Supports two modes:
- JSON response (default): returns all chunks.
- NDJSON streaming (configurable or via `?stream=1`): streams page/section-
    aware chunks incrementally for very large PDFs.

Enhancements:
- Section-aware chunking that respects page headers and headings.
- Table normalization to CSV with `[table id=...]` markers propagated into
    chunk metadata as `table_id`.

This service does not index or persist; the API Gateway registers chunks with
the retrieval service.
"""

from __future__ import annotations

import datetime
import hashlib
import io
import mimetypes
import os
import re
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langdetect import detect  # type: ignore

from shared.models import DocChunk, DocMetadata
from shared.settings import Settings
from shared.tracing import install_fastapi_tracing, span, tracer

from .chunkers import chunk_text, iter_section_chunks
from .readers import parse_document, stream_pdf_pages

app = FastAPI(title="Ingest Service", version="0.1.0")
_settings = Settings()
install_fastapi_tracing(app, service_name="ingest")


@app.post(
    "/ingest",
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
async def ingest(request: Request):
    """Process an uploaded document into a set of chunks.

    This handler manually parses a multipart/form-data request to
    extract the uploaded file. Avoiding FastAPI's built-in UploadFile
    dependency eliminates the need for the python-multipart library.

    Returns:
        A JSON payload containing a generated document ID and a list of
        chunks. Each chunk is represented as a plain dictionary rather
        than a Pydantic model to simplify serialisation.
    """
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        raise HTTPException(
            status_code=400, detail="Content-Type must be multipart/form-data"
        )
    # Extract boundary from header and sanitize quotes/extra params
    boundary_marker = "boundary="
    if boundary_marker not in content_type:
        raise HTTPException(status_code=400, detail="Missing multipart boundary")
    boundary = (
        content_type.split(boundary_marker, 1)[1].split(";", 1)[0].strip().strip('"')
    )

    body = await request.body()
    file_bytes = None
    filename = "uploaded"
    part_mime = None
    # Split body by boundary delimiter, ignoring preamble/epilogue
    parts = body.split(b"--" + boundary.encode())
    for part in parts:
        if b"Content-Disposition" in part and b'name="file"' in part:
            header, _, data_section = part.partition(b"\r\n\r\n")
            # Remove trailing CRLF (the part ends with \r\n)
            data = data_section.rsplit(b"\r\n", 1)[0]
            header_str = header.decode(errors="ignore")
            # Extract filename robustly from Content-Disposition
            m_fn = re.search(r'filename\*?=(?:"([^"]+)"|([^;\r\n]+))', header_str, re.I)
            if m_fn:
                raw_fn = (m_fn.group(1) or m_fn.group(2)).strip()
                filename = os.path.basename(raw_fn)
            # Prefer part Content-Type if present
            m_ct = re.search(r"Content-Type:\s*([^\r\n;]+)", header_str, re.I)
            if m_ct:
                part_mime = m_ct.group(1).strip()
            file_bytes = data
            break
    if not file_bytes:
        raise HTTPException(status_code=400, detail="No file part provided")
    # Derive content type and checksum
    mime = part_mime or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    checksum = hashlib.sha256(file_bytes).hexdigest()
    created_at = datetime.datetime.utcnow().isoformat()

    # Determine streaming preference
    query = dict(request.query_params)
    stream_requested = query.get("stream", "0") in {"1", "true", "yes"}
    is_pdf = filename.lower().endswith(".pdf")

    # Parse full text (non-stream) path
    def json_response():
        with span("ingest.parse_and_chunk", filename=filename):
            raw_text = parse_document(file_bytes, filename)
            doc_id = str(uuid.uuid4())
            # detect language once per document; ignore errors
            try:
                lang = detect(raw_text)[:2]
            except Exception:
                lang = None
            # Prefer section-aware chunking if enabled
            if _settings.chunk_section_aware_enabled:
                chunks_meta = list(
                    iter_section_chunks(
                        raw_text,
                        max_tokens=_settings.chunk_max_tokens,
                        respect_pages=_settings.chunk_respect_pages,
                        respect_headings=_settings.chunk_respect_headings,
                    )
                )
                doc_chunks: list[DocChunk] = []
                for idx, (text, page, section, table_id) in enumerate(chunks_meta):
                    meta = DocMetadata(
                        source=filename,
                        page=page,
                        section=section,
                        table_id=table_id,
                        lang=lang,
                        content_type=mime,
                        checksum=checksum,
                        created_at=created_at,
                    )
                    doc_chunks.append(
                        DocChunk(
                            id=f"{doc_id}_{idx}", doc_id=doc_id, text=text, meta=meta
                        )
                    )
            else:
                parts = chunk_text(raw_text, max_tokens=_settings.chunk_max_tokens)
                doc_chunks = []
                for idx, text in enumerate(parts):
                    meta = DocMetadata(
                        source=filename,
                        lang=lang,
                        content_type=mime,
                        checksum=checksum,
                        created_at=created_at,
                    )
                    doc_chunks.append(
                        DocChunk(
                            id=f"{doc_id}_{idx}", doc_id=doc_id, text=text, meta=meta
                        )
                    )
        return JSONResponse(
            content={"doc_id": doc_id, "chunks": [c.dict() for c in doc_chunks]}
        )

    # Streaming NDJSON path for large PDFs
    def ndjson_stream():
        # Keep the span open until the stream finishes
        stream_span = tracer.start_span("ingest.stream_pdf", filename=filename)
        # Important: enter the span so it actually starts recording
        stream_span.__enter__()
        doc_id = str(uuid.uuid4())
        # Detect lang incrementally (roughly) by first page text
        try:
            g = stream_pdf_pages(file_bytes)
            first_page = next(g)
            try:
                lang_local = detect(first_page[1])[:2]
            except Exception:
                lang_local = None

            # Create a generator that yields the first page then the rest
            def page_gen():
                yield first_page
                for item in g:
                    yield item
        except StopIteration:
            # Close span before fallback
            stream_span.__exit__(None, None, None)
            return json_response()

        def iter_ndjson():
            idx = 0
            for page_num, page_text in page_gen():
                # Feed page_text through section-aware chunker with pages respected
                for text, _, section, table_id in iter_section_chunks(
                    page_text,
                    max_tokens=_settings.chunk_max_tokens,
                    respect_pages=True,
                    respect_headings=_settings.chunk_respect_headings,
                ):
                    meta = DocMetadata(
                        source=filename,
                        page=page_num,
                        section=section,
                        table_id=table_id,
                        lang=lang_local,
                        content_type=mime,
                        checksum=checksum,
                        created_at=created_at,
                    )
                    chunk = DocChunk(
                        id=f"{doc_id}_{idx}", doc_id=doc_id, text=text, meta=meta
                    )
                    idx += 1
                    yield (chunk, doc_id)

        def encode_ndjson():
            try:
                yield ("{" + f'"doc_id":"{doc_id}"' + "}\n").encode()
                for chunk, _doc in iter_ndjson():
                    yield (chunk.json() + "\n").encode()
            finally:
                # End the stream span when the stream is done
                stream_span.__exit__(None, None, None)

        return StreamingResponse(encode_ndjson(), media_type="application/x-ndjson")

    should_stream = False
    if is_pdf and (stream_requested or _settings.ingest_streaming_enabled):
        # If auto-streaming, only when page count exceeds threshold
        if stream_requested:
            should_stream = True
        else:
            try:
                from PyPDF2 import PdfReader  # type: ignore

                pages = len(PdfReader(io.BytesIO(file_bytes)).pages)
                should_stream = pages >= _settings.ingest_stream_pdf_min_pages
            except Exception:
                # If page counting fails, fall back to non-stream
                should_stream = False
    if should_stream:
        return ndjson_stream()
    return json_response()
