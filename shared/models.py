"""Pydantic data models shared across services.

These models define the core data structures used in the Intelligent Content
Analyzer. By centralising them in a shared module we ensure that all
micro‑services agree on the shape of the data they send and receive.

The definitions here are intentionally simple and typed. Additional fields can
be added over time without breaking existing consumers as long as sensible
defaults are provided.
"""

from __future__ import annotations

from typing import List, Optional, Any

from pydantic import BaseModel, Field


class DocMetadata(BaseModel):
    """Metadata associated with a document chunk.

    Attributes:
        source: The original filename or external identifier for the source.
        page: The page number within the source document, if applicable.
        section: A logical section or heading name, if known.
        lang: The language code of the text, e.g. 'en' or 'ko'.
    """

    source: str
    page: Optional[int] = None
    section: Optional[str] = None
    lang: Optional[str] = None


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
    """

    query: str
    top_k: int = 10
    hybrid: bool = True


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


class EvaluateRequest(BaseModel):
    """Request body for answer evaluation.

    Contains the original question, the generated answer, and the
    document chunks used as sources.
    """

    question: str
    answer: str
    sources: List[DocChunk]


class EvaluateResponse(BaseModel):
    """Response from the evaluation service.

    Scores are normalised to the range [0,1].
    """

    factuality: float
    relevance: float
    completeness: float
    comments: Optional[str] = None