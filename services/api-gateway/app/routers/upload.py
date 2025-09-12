"""Upload router for the API gateway.

Handles document uploads and invokes the ingestion logic to produce
chunks. The gateway does not persist the chunks; instead it
immediately registers them with the in‑memory index of the retrieval
service. This makes subsequent queries against the document possible
without standing up a message queue or database.
"""

from __future__ import annotations

import uuid
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

from services.ingest.app.readers import parse_document
from services.ingest.app.chunkers import chunk_text
from services.retrieval.app.main import INDEX  # type: ignore
from shared.models import DocChunk, DocMetadata


router = APIRouter()


@router.post("/upload_document")
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
        raise HTTPException(status_code=400, detail="Content-Type must be multipart/form-data")
    boundary_marker = "boundary="
    if boundary_marker not in content_type:
        raise HTTPException(status_code=400, detail="Missing multipart boundary")
    boundary = content_type.split(boundary_marker, 1)[1]
    body = await request.body()
    file_bytes = None
    filename = "uploaded"
    # Split body by boundary delimiter, ignoring preamble/epilogue
    delimiter = b"--" + boundary.encode()
    parts = body.split(delimiter)
    for part in parts:
        if b"Content-Disposition" in part and b"name=\"file\"" in part:
            header, _, data_section = part.partition(b"\r\n\r\n")
            # Remove trailing CRLF (the part ends with CRLF)
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
    chunks: list[DocChunk] = []
    for idx, chunk in enumerate(text_chunks):
        meta = DocMetadata(source=filename)
        chunks.append(DocChunk(id=f"{doc_id}_{idx}", doc_id=doc_id, text=chunk, meta=meta))
    # Register chunks in retrieval index
    INDEX.extend(chunks)
    return JSONResponse({"doc_id": doc_id, "num_chunks": len(chunks)})