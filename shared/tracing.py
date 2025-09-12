"""Lightweight tracing utilities.

In a production deployment you would wire these hooks up to a tracing
platform such as Langfuse, OpenTelemetry or Zipkin. Here we provide a
minimal shim that behaves as a no‑op when tracing is disabled. This
allows the rest of the application code to call `tracer.start_span`
without worrying about the underlying implementation.
"""

from contextlib import contextmanager
from typing import Any, Iterator
import os


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
    """Simple tracer facade.

    If the environment variable `LANGFUSE_ENABLED` is set to 'true', this
    could be extended to integrate with Langfuse. Currently it always
    returns no‑op spans.
    """

    def start_span(self, name: str, **kwargs: Any) -> _Span:
        # Always return a span; instrumentation would be inserted here.
        return _Span(name, **kwargs)


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