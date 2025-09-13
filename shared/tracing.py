"""Tracing utilities with optional Langfuse integration.

By default this module provides a lightweight span context manager that
prints nothing (no-op). If `LANGFUSE_ENABLED=true` and the `langfuse`
Python SDK is installed and configured via environment variables, spans
are forwarded to Langfuse. Errors in tracing never affect application
logic; we fail-soft to a no-op.

This module also exposes lightweight helpers for observability events
(`log_event`) and crude token estimation (`estimate_tokens`).
"""

import os
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

try:
    from langfuse import Langfuse  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Langfuse = None  # type: ignore


def _now_ms() -> int:
    return int(time.time() * 1000)


class _Span:
    """A no‑op span used when tracing is disabled."""

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name

    def __enter__(self) -> "_Span":
        # In a real tracer you might record start time or allocate an ID here
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # In a real tracer you would record the end time and any errors
        return None


class Tracer:
    """Tracer facade with pluggable backends (no-op or Langfuse)."""

    def __init__(self) -> None:
        self._backend = os.getenv("TRACING_BACKEND", "langfuse").lower()
        self._enabled = os.getenv("LANGFUSE_ENABLED", "false").lower() in (
            "1",
            "true",
            "yes",
        )
        self._client = None
        if self._enabled and self._backend == "langfuse" and Langfuse is not None:
            try:
                # Langfuse expects public/secret keys; we allow host override
                public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
                secret_key = os.getenv("LANGFUSE_SECRET_KEY")
                host = os.getenv("LANGFUSE_HOST")
                if public_key and secret_key:
                    self._client = Langfuse(
                        public_key=public_key, secret_key=secret_key, host=host
                    )
            except Exception:
                self._client = None

    def start_span(self, name: str, **kwargs: Any) -> _Span:
        if self._client is None:
            return _Span(name, **kwargs)
        return _LangfuseSpan(self._client, name, **kwargs)


tracer = Tracer()


@contextmanager
def span(name: str, **kwargs: Any) -> Iterator[_Span]:
    """Context manager wrapper around the tracer's start_span method.

    Usage:
        with span("my_operation"):
            # do work
    """
    s = tracer.start_span(name, **kwargs)
    try:
        yield s
    finally:
        # __exit__ is invoked when leaving context
        s.__exit__(None, None, None)


class _LangfuseSpan(_Span):  # pragma: no cover - optional dependency
    def __init__(self, client: Any, name: str, **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._client = client
        self._trace = None
        self._span = None
        self._start_ms = _now_ms()
        self._kwargs = kwargs

    def __enter__(self) -> "_LangfuseSpan":
        try:
            # Create or reuse a single trace per process; fall back if unavailable
            if hasattr(self._client, "trace"):
                self._trace = self._client.trace(
                    name=os.getenv("TRACE_NAME", "ica-trace")
                )
                if hasattr(self._trace, "span"):
                    self._span = self._trace.span(name=self.name, input=self._kwargs)
            elif hasattr(self._client, "span"):
                self._span = self._client.span(name=self.name, input=self._kwargs)
        except Exception:
            self._trace = None
            self._span = None
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            duration_ms = max(1, _now_ms() - self._start_ms)
            if self._span and hasattr(self._span, "end"):
                # Some SDKs support explicit end
                self._span.end(
                    output={
                        "error": str(exc) if exc else None,
                        "duration_ms": duration_ms,
                    }
                )
            elif self._trace and hasattr(self._trace, "span"):
                # Create a terminal span if not explicitly tracked
                self._trace.span(
                    name=f"{self.name}:end", input={"duration_ms": duration_ms}
                )
        except Exception:
            pass


def log_event(
    name: str, payload: Optional[dict] = None, correlation_id: Optional[str] = None
) -> None:
    """Emit a short-lived structured event span for observability.

    Args:
        name: Logical event name, e.g. "Retrieval", "Generation", "Evaluation", "Feedback".
        payload: Arbitrary JSON-serializable dict with event data.
        correlation_id: Optional ID to stitch events across services.
    """
    meta = dict(payload or {})
    if correlation_id:
        meta["correlation_id"] = correlation_id
    with span(f"event.{name}", **meta):
        # no body; end immediately
        pass


def estimate_tokens(text: str) -> int:
    """Very rough subword token estimate (for logging only)."""
    if not text:
        return 0
    # Approximate: 1 token ~= 4 chars for English-like text
    return max(1, int(len(text) / 4))
