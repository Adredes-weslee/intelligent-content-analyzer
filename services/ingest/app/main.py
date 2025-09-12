"""Entry point for the ingest microservice.

This FastAPI application exposes an endpoint to upload a document and
split it into chunks. Each returned chunk includes basic metadata to
facilitate downstream indexing and retrieval. In a full deployment the
ingest service would persist these chunks to a database or message
queue; here we simply return them in the response.
"""

from __future__ import annotations

import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from shared.models import DocChunk, DocMetadata
from .readers import parse_document
from .chunkers import chunk_text

app = FastAPI(title="Ingest Service", version="0.1.0")


@app.post("/ingest")
async def ingest(request: Request) -> JSONResponse:
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
        raise HTTPException(status_code=400, detail="Content-Type must be multipart/form-data")
    # Extract boundary from header (e.g. boundary=----WebKitFormBoundary12345)
    boundary_marker = "boundary="
    if boundary_marker not in content_type:
        raise HTTPException(status_code=400, detail="Missing multipart boundary")
    boundary = content_type.split(boundary_marker, 1)[1]
    body = await request.body()
    file_bytes = None
    filename = "uploaded"
    # Split body by boundary delimiter, ignoring preamble/epilogue
    parts = body.split(b"--" + boundary.encode())
    for part in parts:
        if b"Content-Disposition" in part and b"name=\"file\"" in part:
            header, _, data_section = part.partition(b"\r\n\r\n")
            # Remove trailing CRLF (the part ends with \r\n)
            data = data_section.rsplit(b"\r\n", 1)[0]
            header_str = header.decode(errors="ignore")
            # Extract filename if present
            if "filename=" in header_str:
                filename = header_str.split("filename=")[-1].strip().strip('"')
            file_bytes = data
            break
    if not file_bytes:
        raise HTTPException(status_code=400, detail="No file part provided")
    raw_text = parse_document(file_bytes, filename)
    doc_id = str(uuid.uuid4())
    text_chunks = chunk_text(raw_text)
    doc_chunks: list[DocChunk] = []
    for idx, chunk in enumerate(text_chunks):
        meta = DocMetadata(source=filename)
        doc_chunks.append(
            DocChunk(id=f"{doc_id}_{idx}", doc_id=doc_id, text=chunk, meta=meta)
        )
    return JSONResponse(
        content={
            "doc_id": doc_id,
            "chunks": [c.dict() for c in doc_chunks],
        }
    )