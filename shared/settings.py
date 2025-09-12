"""Centralised application configuration.

Settings are loaded from environment variables or the `.env` file in the
project root. Using pydantic's BaseSettings provides convenient parsing
and type checking. Each microservice should instantiate its own Settings
instance when starting up.
"""

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    model_provider: str = Field("GEMINI", env="MODEL_PROVIDER")
    embedding_model: str = Field("bge-m3", env="EMBEDDING_MODEL")
    vector_store: str = Field("FAISS", env="VECTOR_STORE")
    reranker: str = Field("bge-reranker", env="RERANKER")
    language_mode: str = Field("multilingual", env="LANGUAGE_MODE")
    cache_enabled: bool = Field(True, env="CACHE_ENABLED")
    confidence_threshold: float = Field(0.5, env="CONFIDENCE_THRESHOLD")

    # Logging/observability
    langfuse_enabled: bool = Field(True, env="LANGFUSE_ENABLED")
    langfuse_api_key: str = Field("", env="LANGFUSE_API_KEY")
    langfuse_host: str = Field("", env="LANGFUSE_HOST")

    class Config:
        env_file = ".env"
        case_sensitive = False