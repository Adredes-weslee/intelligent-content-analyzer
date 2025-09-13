"""Pydantic data models shared across services.

These models define the core data structures used in the Intelligent Content
Analyzer. By centralising them in a shared module we ensure that all
micro‑services agree on the shape of the data they send and receive.

The definitions here are intentionally simple and typed. Additional fields can
be added over time without breaking existing consumers as long as sensible
defaults are provided.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocMetadata(BaseModel):
    """Metadata associated with a document chunk.

    Attributes:
        source: The original filename or external identifier for the source.
        page: The page number within the source document, if applicable.
        section: A logical section or heading name, if known.
        lang: The language code of the text, e.g. 'en' or 'ko'.
        title: Optional document or section title.
        content_type: MIME type or simplified format (e.g. 'application/pdf').
        checksum: Optional content checksum (e.g. SHA-256) for idempotency.
        source_uri: Optional URI/path to the original source.
        figure_label: Optional figure label if the chunk describes an image.
        table_id: Optional table identifier if extracted from a table.
        created_at: Optional ISO timestamp when ingested.
        updated_at: Optional ISO timestamp when last updated.
        extra: Free-form metadata map for future extensibility.
    """

    source: str
    page: Optional[int] = None
    section: Optional[str] = None
    lang: Optional[str] = None
    title: Optional[str] = None
    content_type: Optional[str] = None
    checksum: Optional[str] = None
    source_uri: Optional[str] = None
    figure_label: Optional[str] = None
    table_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class DocChunk(BaseModel):
    """A slice of text extracted from a source document.

    Each chunk should represent a reasonably small unit of knowledge such that
    the downstream language model can reason about it effectively. Chunks are
    immutable once created.
    """

    id: str = Field(..., description="Unique identifier for the chunk")
    doc_id: str = Field(..., description="Identifier of the parent document")
    text: str = Field(..., description="The textual content of the chunk")
    meta: DocMetadata = Field(..., description="Associated metadata for the chunk")


class EmbedRequest(BaseModel):
    """Request body for embedding service.

    It carries a list of document chunks to be converted into vectors.
    """

    chunks: List[DocChunk]


class EmbedResponse(BaseModel):
    """Response from the embedding service.

    Each vector corresponds to an input chunk by position.
    """

    vectors: List[List[float]]
    model: str


class RetrieveRequest(BaseModel):
    """Request body for the retrieval service.

    Attributes:
        query: The natural language question to search for.
        top_k: The number of results to return.
        hybrid: Whether to combine BM25 and dense retrieval methods.
        correlation_id: Optional ID used to stitch observability events
            across services for a single user request.
    """

    query: str
    top_k: int = 10
    hybrid: bool = True
    correlation_id: Optional[str] = None


class RetrieveResult(BaseModel):
    """Individual search hit returned from the retriever.

    The score field is a general relevance score. Additional algorithm
    specific scores can be attached on a best‑effort basis.
    """

    chunk: DocChunk
    score: float
    bm25: Optional[float] = None
    dense: Optional[float] = None


class RetrieveResponse(BaseModel):
    """Response from the retrieval service.

    A list of hits in descending order of relevance.
    """

    hits: List[RetrieveResult]
    diagnostics: Optional[Any] = None


class QARequest(BaseModel):
    """Request for the question answering service.

    Attributes:
        question: The question posed by the user.
        k: Number of context chunks to use when answering.
        use_rerank: Whether to apply re‑ranking on retrieved results.
        reranker: Optional name of the reranker model.
    """

    question: str
    k: int = 5
    use_rerank: bool = True
    reranker: Optional[str] = None


class Citation(BaseModel):
    """A single citation attached to a QA response."""

    doc_id: str
    page: Optional[int] = None
    section: Optional[str] = None


class QAResponse(BaseModel):
    """Response from the question answering service.

    Contains the answer text, a list of citations backing the answer,
    an overall confidence score (0–1), and optional diagnostic info.
    """

    answer: str
    citations: List[Citation]
    confidence: float
    diagnostics: Optional[Any] = None


class SummarizeRequest(BaseModel):
    """Request body for document summarization.

    Provide either a doc_id (for reference) and/or explicit chunks.
    The gateway typically supplies chunks to avoid tight coupling.
    """

    doc_id: Optional[str] = None
    # Optional correlation id to stitch events across services for one request
    correlation_id: Optional[str] = None
    chunks: List[DocChunk]


class SummarizeResponse(BaseModel):
    """Response from summarization.

    Contains a concise summary, key points, optional citations, and diagnostics.
    """

    summary: str
    key_points: List[str] = []
    citations: List[Citation] = []
    diagnostics: Optional[Any] = None


class GenerateRequest(BaseModel):
    """Structured request for the generation service.

    Passes the user question and the explicit context chunks to be used as
    evidence. This avoids embedding the context into a free‑form prompt and
    keeps contracts between services type‑safe.
    """

    question: str
    context_chunks: List[DocChunk]
    # Optional correlation id to stitch events across services for one request
    correlation_id: Optional[str] = None


class GenerateResponse(BaseModel):
    """Optional explicit response model for the generation service.

    Mirrors `QAResponse` to allow the generator to be used standalone, while
    keeping the canonical QA response type for the gateway.
    """

    answer: str
    citations: List[Citation]
    diagnostics: Optional[Any] = None


class EvaluateRequest(BaseModel):
    """Request body for answer evaluation.

    Contains the original question, the generated answer, and the
    document chunks used as sources. Optionally include the full
    retrieved hits (with scores) to enable context‑relevance metrics.
    """

    question: str
    answer: str
    sources: List[DocChunk]
    # Optional: provide scored hits to compute context relevance ratio
    hits: Optional[List[RetrieveResult]] = None


class EvaluateResponse(BaseModel):
    """Response from the evaluation service.

    Scores are normalised to the range [0,1] unless otherwise noted.
    """

    factuality: float
    relevance: float
    completeness: float
    # Extended judge rubric (optional fields):
    # - faithfulness: fraction of answer statements grounded in sources [0,1]
    faithfulness: Optional[float] = None
    # - answer_relevance_1_5: how well the answer addresses the question [1–5]
    answer_relevance_1_5: Optional[float] = None
    # - context_relevance_ratio: fraction of retrieved chunks relevant [0,1]
    context_relevance_ratio: Optional[float] = None
    comments: Optional[str] = None


class FeedbackRequest(BaseModel):
    """User feedback on a QA response.

    Attributes:
        correlation_id: Correlates feedback to a single QA trace.
        question: The original question text.
        answer: The answer text shown to the user (optional).
        rating: "up" or "down" (thumbs).
        comment: Optional free-text explanation.
    """

    correlation_id: Optional[str] = None
    question: str
    answer: Optional[str] = None
    rating: str
    comment: Optional[str] = None


class FeedbackAck(BaseModel):
    """Acknowledgement for feedback submission."""

    ok: bool = True
    stored: bool = True
