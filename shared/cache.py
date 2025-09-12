"""Simple cache abstraction.

This module abstracts access to a key/value cache. In production this could
be backed by Redis or another in‑memory store. For testing and local
development we fall back to a process‑local dictionary. Keys and values
are serialised using JSON where necessary.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

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


def get_default_cache() -> Cache:
    """Initialise a default cache based on environment configuration."""
    url = os.getenv("CACHE_REDIS_URL")
    enabled = os.getenv("CACHE_ENABLED", "true").lower() in ("1", "true", "yes")
    return Cache(url if enabled else None)