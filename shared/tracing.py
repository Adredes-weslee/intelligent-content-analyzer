"""Tracing utilities with optional Langfuse integration.

By default this module provides a lightweight span context manager that
prints nothing (no-op). If `LANGFUSE_ENABLED=true` and the `langfuse`
Python SDK is installed and configured via environment variables, spans
are forwarded to Langfuse. Errors in tracing never affect application
logic; we fail-soft to a no-op.

This module also exposes lightweight helpers for observability events
(`log_event`) and crude token estimation (`estimate_tokens`).
"""

import contextvars
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

from shared.settings import Settings

_settings = Settings()

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


_current_trace = contextvars.ContextVar("ica.current_trace", default=None)


class Tracer:
    """Tracer facade with pluggable backends (no-op or Langfuse)."""

    def __init__(self) -> None:
        self._backend = _settings.tracing_backend.lower()

        # Use explicit OS env if set; otherwise use Settings (.env)
        self._enabled = bool(_settings.langfuse_enabled)

        self._client = None
        if self._enabled and self._backend == "langfuse" and Langfuse is not None:
            try:
                public_key = _settings.langfuse_public_key
                secret_key = _settings.langfuse_secret_key
                host = _settings.langfuse_host
                if public_key and secret_key:
                    self._client = Langfuse(
                        public_key=public_key, secret_key=secret_key, host=host
                    )
            except Exception:
                self._client = None

    def start_trace(
        self, name: str, input: Optional[dict] = None, user_id: Optional[str] = None
    ):
        if self._client is None:
            return None
        try:
            tr = self._client.trace(name=name, input=input or {}, user_id=user_id)
            _current_trace.set(tr)
            return tr
        except Exception:
            return None

    def end_trace(self, output: Optional[dict] = None):
        tr = _current_trace.get()
        if tr and hasattr(tr, "end"):
            try:
                tr.end(output=output or {})
            except Exception:
                pass
        _current_trace.set(None)

    def start_span(self, name: str, **kwargs: Any) -> _Span:
        if self._client is None:
            return _Span(name, **kwargs)
        # Prefer attaching spans to the current request trace if present
        tr = _current_trace.get()
        if tr is not None:
            return _LangfuseSpan(self._client, name, parent_trace=tr, **kwargs)
        return _LangfuseSpan(self._client, name, **kwargs)


tracer = Tracer()


def install_fastapi_tracing(app, service_name: str = "ingest") -> None:
    """Install middleware to auto-create a Langfuse trace per HTTP request."""
    try:
        from fastapi import Request
    except Exception:
        return  # FastAPI not installed

    @app.middleware("http")
    async def _trace_middleware(request: "Request", call_next: Callable):
        # Build a concise input payload (avoid full headers/body for PII)
        route = getattr(request.scope, "route", None)
        route_path = getattr(route, "path", request.url.path)
        tracer.start_trace(
            name=f"{service_name} {request.method} {route_path}",
            input={
                "method": request.method,
                "path": route_path,
                "query": str(request.url.query) if request.url.query else "",
            },
        )
        try:
            with span("http.request"):
                response = await call_next(request)
            return response
        except Exception as e:
            # Record an error span; trace will end below
            with span("http.error", error=str(e)):
                pass
            raise
        finally:
            tracer.end_trace(
                output={
                    "status": getattr(
                        locals().get("response", None), "status_code", None
                    )
                }
            )


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
    def __init__(
        self, client: Any, name: str, parent_trace: Any | None = None, **kwargs: Any
    ) -> None:
        super().__init__(name, **kwargs)
        self._client = client
        self._trace = None
        # Use the current request trace if provided by Tracer.start_span
        self._trace = parent_trace
        self._span = None
        self._start_ms = _now_ms()
        self._kwargs = kwargs

    def __enter__(self) -> "_LangfuseSpan":
        try:
            # If no current request trace, create a lightweight one so span still records
            if self._trace is None and hasattr(self._client, "trace"):
                self._trace = self._client.trace(name=_settings.trace_name)
            if self._trace is not None and hasattr(self._trace, "span"):
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
