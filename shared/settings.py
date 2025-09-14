"""Centralised application configuration.

Settings are loaded from environment variables or the `.env` file in the
project root. Using pydantic's BaseSettings provides convenient parsing
and type checking. Each microservice should instantiate its own Settings
instance when starting up. This module also centralises feature flags such
as offline mode and cache configuration so behaviour is consistent across
services.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Upstream service URLs (enable HTTP microservices mode when set)
    ingest_url: str | None = Field(
        default=None, validation_alias=AliasChoices("INGEST_URL", "ingest_url")
    )
    retrieval_url: str | None = Field(
        default=None, validation_alias=AliasChoices("RETRIEVAL_URL", "retrieval_url")
    )
    embeddings_url: str | None = Field(
        default=None, validation_alias=AliasChoices("EMBEDDINGS_URL", "embeddings_url")
    )
    llm_generate_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_GENERATE_URL", "llm_generate_url"),
    )
    evaluation_url: str | None = Field(
        default=None, validation_alias=AliasChoices("EVALUATION_URL", "evaluation_url")
    )
    # UI/CORS (optional)
    streamlit_app_origin: str | None = Field(
        default=None,
        validation_alias=AliasChoices("STREAMLIT_APP_ORIGIN", "streamlit_app_origin"),
    )

    # Gemini
    gemini_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "gemini_api_key")
    )
    gemini_fast_model: str = Field(
        default="gemini-2.5-flash",
        validation_alias=AliasChoices("GEMINI_FAST_MODEL", "GEMINI_MODEL"),
    )
    gemini_reasoning_model: str = Field(
        default="gemini-2.5-pro", validation_alias="GEMINI_REASONING_MODEL"
    )

    # Embeddings
    embedding_model: str = Field(
        default="gemini-embedding-001",
        validation_alias=AliasChoices("EMBEDDING_MODEL", "embedding_model"),
    )
    # Accept both EMBEDDING_DIM (tests/Render) and GEMINI_EMBEDDING_DIM (legacy)
    embedding_dim: int = Field(
        default=768,
        validation_alias=AliasChoices(
            "EMBEDDING_DIM", "GEMINI_EMBEDDING_DIM", "embedding_dim"
        ),
    )

    # Retrieval/vector store
    vector_store: str = Field("FAISS", env="VECTOR_STORE")
    faiss_index_path: str = Field("./data/faiss.index", env="FAISS_INDEX_PATH")
    faiss_metric: str = Field("ip", env="FAISS_METRIC")  # "ip" or "l2"
    faiss_normalize: bool = Field(True, env="FAISS_NORMALIZE")
    doc_map_path: str = Field("data/doc_map.json", env="DOC_MAP_PATH")
    enable_upsert: bool = Field(True, env="ENABLE_UPSERT")

    # Hybrid/dense candidates
    dense_candidates: int = Field(50, env="DENSE_CANDIDATES")

    # Reranking
    reranker: str = Field("bge-reranker", env="RERANKER")
    reranker_backend: str = Field("cross-encoder", env="RERANKER_BACKEND")
    reranker_model: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-6-v2", env="RERANKER_MODEL"
    )
    rerank_threshold: float = Field(0.2, env="RERANK_THRESHOLD")

    # Language and query refinement
    language_mode: str = Field("multilingual", env="LANGUAGE_MODE")
    query_refine_enabled: bool = Field(True, env="QUERY_REFINE_ENABLED")

    # Caching & performance
    cache_enabled: bool = Field(True, env="CACHE_ENABLED")
    confidence_threshold: float = Field(0.5, env="CONFIDENCE_THRESHOLD")
    answer_cache_ttl_seconds: int = Field(7 * 24 * 3600, env="ANSWER_CACHE_TTL_SECONDS")
    summary_cache_ttl_seconds: int = Field(
        7 * 24 * 3600, env="SUMMARY_CACHE_TTL_SECONDS"
    )
    semantic_cache_threshold: float = Field(0.92, env="SEMANTIC_CACHE_THRESHOLD")
    semantic_cache_max_entries: int = Field(1000, env="SEMANTIC_CACHE_MAX_ENTRIES")

    # Unified toggles/infra
    offline_mode: bool = Field(False, env="OFFLINE_MODE")
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")

    # Logging/observability
    langfuse_enabled: bool = Field(True, env="LANGFUSE_ENABLED")
    langfuse_api_key: str = Field("", env="LANGFUSE_API_KEY")
    langfuse_host: str = Field("", env="LANGFUSE_HOST")
    # Support public/secret key pair if available
    langfuse_public_key: str = Field("", env="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field("", env="LANGFUSE_SECRET_KEY")
    # Tracing config
    tracing_backend: str = Field("langfuse", env="TRACING_BACKEND")
    trace_name: str = Field("ica-trace", env="TRACE_NAME")

    # Ingestion and parsing limits
    ingest_max_file_mb: int = Field(150, env="INGEST_MAX_FILE_MB")
    ingest_max_pages: int = Field(500, env="INGEST_MAX_PAGES")
    ingest_streaming_enabled: bool = Field(False, env="INGEST_STREAMING_ENABLED")
    # If a PDF has at least this many pages, prefer streaming NDJSON output
    ingest_stream_pdf_min_pages: int = Field(20, env="INGEST_STREAM_PDF_MIN_PAGES")
    chunk_max_tokens: int = Field(200, env="CHUNK_MAX_TOKENS")

    # Section-aware chunking and parsing toggles
    chunk_section_aware_enabled: bool = Field(True, env="CHUNK_SECTION_AWARE_ENABLED")
    chunk_respect_pages: bool = Field(True, env="CHUNK_RESPECT_PAGES")
    chunk_respect_headings: bool = Field(True, env="CHUNK_RESPECT_HEADINGS")
    normalize_tables_to_csv: bool = Field(True, env="NORMALIZE_TABLES_TO_CSV")

    # PDF/image handling and OCR via Gemini
    gemini_multimodal_enabled: bool = Field(True, env="GEMINI_MULTIMODAL_ENABLED")
    pdf_render_images: bool = Field(True, env="PDF_RENDER_IMAGES")

    # Summarization and evaluation toggles
    summarizer_max_chunks: int = Field(12, env="SUMMARIZER_MAX_CHUNKS")
    eval_llm_enabled: bool = Field(True, env="EVAL_LLM_ENABLED")

    # Rate limiting
    rate_limit_enabled: bool = Field(True, env="RATE_LIMIT_ENABLED")
    rate_limit_per_minute: int = Field(5, env="RATE_LIMIT_PER_MINUTE")
