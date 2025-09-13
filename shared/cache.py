"""Cache utilities: key/value cache plus semantic QA cache helpers.

This module provides:
- A simple key/value cache interface backed by Redis when available or an
    in‑process dict for local dev/tests.
- Semantic QA cache helpers, including a content fingerprint function and a
    lightweight in‑process semantic cache for paraphrase lookups.

In production, consider backing the semantic cache with a proper vector
store or Redis modules. Here we keep dependencies minimal and rely on
the embeddings service for vectorisation.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from typing import Any, Iterable, List, Optional, Tuple

try:
    import redis  # type: ignore
except ImportError:
    redis = None  # type: ignore


class Cache:
    """A wrapper around Redis with a fallback to an in‑memory dict."""

    def __init__(self, url: Optional[str] = None) -> None:
        self._local_store: dict[str, Any] = {}
        self._client = None
        if redis is not None and url:
            try:
                self._client = redis.Redis.from_url(url, decode_responses=True)
            except Exception:
                # Fall back to local store if Redis is unavailable
                self._client = None

    def get(self, key: str) -> Optional[Any]:
        if self._client:
            try:
                val = self._client.get(key)
                return json.loads(val) if val is not None else None
            except Exception:
                return None
        return self._local_store.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if self._client:
            try:
                val = json.dumps(value)
                if ttl is not None:
                    self._client.set(key, val, ex=ttl)
                else:
                    self._client.set(key, val)
                return
            except Exception:
                # fall back to local store on error
                pass
        self._local_store[key] = value


INDEX_VERSION_KEY = "qa:index_version"


def get_index_version() -> int:
    """Return a monotonically increasing index version for cache namespacing."""
    try:
        cache = get_default_cache()
        val = cache.get(INDEX_VERSION_KEY)
        if val is None:
            return 0
        try:
            return int(val if not isinstance(val, dict) else val.get("v", 0))
        except Exception:
            return 0
    except Exception:
        return 0


def bump_index_version() -> int:
    """Increment and persist the index version to invalidate QA caches."""
    try:
        cache = get_default_cache()
        cur = get_index_version()
        nxt = cur + 1
        cache.set(INDEX_VERSION_KEY, nxt)
        return nxt
    except Exception:
        return 0


# --- Semantic cache helpers -------------------------------------------------


def _deterministic_vector(text: str, dim: int = 128) -> list[float]:
    """Produce a deterministic pseudo-embedding from text.

    Uses SHA256 hash expanded to `dim` floats in [0,1). This is only for
    OFFLINE_MODE determinism and should not be used in production.
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Repeat hash bytes to cover required dimension
    bytes_needed = dim
    buf = (h * ((bytes_needed // len(h)) + 1))[:bytes_needed]
    return [b / 255.0 for b in buf]


def semantic_key(question: str, scope: str = "global", dim: int = 128) -> str:
    """Build a semantic cache key using embeddings of the question text.

    If OFFLINE_MODE is enabled, use a deterministic pseudo-embedding to ensure
    repeatable keys. Otherwise, fall back to hashing normalized text.
    """
    try:
        from shared.settings import Settings  # type: ignore

        s = Settings()
        if s.offline_mode:
            vec = _deterministic_vector(question, dim=dim)
            raw = ":".join(f"{x:.4f}" for x in vec).encode("utf-8")
            h = hashlib.sha256(raw + b"|" + scope.encode("utf-8")).hexdigest()
            return f"semqa:{h}"
    except Exception:
        pass
    normalized = " ".join(question.strip().lower().split())
    h = hashlib.sha256(
        normalized.encode("utf-8") + b"|" + scope.encode("utf-8")
    ).hexdigest()
    return f"semqa:{h}"


# --- Fingerprinting and semantic QA cache ----------------------------------


def content_fingerprint(doc_ids: Iterable[str]) -> str:
    """Return a stable fingerprint for a set of document IDs.

    We hash the sorted unique doc ids to generate a short fingerprint that
    scopes semantic cache hits to the same content set.
    """
    ids = sorted({str(x) for x in doc_ids if x})
    raw = "|".join(ids).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (da * db)


class SemanticQACache:
    """In-process semantic QA cache partitioned by content fingerprint.

    Stores (embedding, normalized_question, fingerprint, cache_key) tuples.
    The actual QA payload is stored in the default cache under `cache_key`.
    """

    def __init__(self, threshold: float, max_entries: int) -> None:
        self.threshold = threshold
        self.max_entries = max_entries
        self._entries: List[Tuple[List[float], str, str, str]] = []
        self._lock = threading.Lock()

    def _embed(self, question: str) -> Optional[List[float]]:
        try:
            # Lazy import to avoid strong coupling
            from services.embeddings.app.embeddings import embed_texts  # type: ignore

            return [float(x) for x in embed_texts([question])[0]]
        except Exception:
            return None

    def add(self, question: str, fingerprint: str, cache_key: str) -> None:
        vec = self._embed(question)
        if not vec:
            return
        with self._lock:
            self._entries.append(
                (vec, question.strip().lower(), fingerprint, cache_key)
            )
            if len(self._entries) > self.max_entries:
                self._entries = self._entries[-self.max_entries :]

    def query(self, question: str, fingerprint: str) -> Optional[str]:
        vec = self._embed(question)
        if not vec:
            return None
        best_key = None
        best_score = 0.0
        with self._lock:
            for v, _q, fp, key in self._entries:
                if fp != fingerprint:
                    continue
                score = _cosine(vec, v)
                if score > best_score:
                    best_score = score
                    best_key = key
        # Pull threshold from settings if possible; otherwise default 0.92
        try:
            from shared.settings import Settings  # type: ignore

            thresh = Settings().semantic_cache_threshold
        except Exception:
            thresh = 0.92
        if best_key and best_score >= thresh:
            return best_key
        return None


_semantic_cache: Optional[SemanticQACache] = None


def get_semantic_cache() -> SemanticQACache:
    """Return a process-wide semantic QA cache singleton."""
    global _semantic_cache
    if _semantic_cache is None:
        try:
            from shared.settings import Settings  # type: ignore

            s = Settings()
            _semantic_cache = SemanticQACache(
                threshold=s.semantic_cache_threshold,
                max_entries=s.semantic_cache_max_entries,
            )
        except Exception:
            _semantic_cache = SemanticQACache(threshold=0.92, max_entries=1000)
    return _semantic_cache


def get_default_cache() -> Cache:
    """Initialise a default cache based on environment configuration.

    Prefers values from shared.settings.Settings; falls back to env vars for
    compatibility. When disabled, returns an in-memory cache.
    """
    try:
        # Local import to avoid circular imports at module load time
        from shared.settings import Settings  # type: ignore

        s = Settings()
        if not s.cache_enabled:
            return Cache(None)

        url = s.redis_url or None
        return Cache(url)
    except Exception:
        # If Settings cannot be loaded, fall back to in-memory cache
        return Cache(None)
