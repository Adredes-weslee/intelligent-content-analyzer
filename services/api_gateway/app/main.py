"""API gateway for the Intelligent Content Analyzer.

This service exposes a unified REST interface to the outside world and
delegates work to the underlying microservices. It hides the
complexity of ingestion, retrieval, generation and evaluation behind
three simple endpoints. The gateway composes synchronous calls to the
Python modules directly rather than making network requests. This
keeps the example small and avoids the need to start all services
during testing. In a real deployment the gateway would send HTTP
requests to the other services.
"""

from __future__ import annotations

from fastapi import FastAPI

from .routers import upload, qa, summary

app = FastAPI(title="API Gateway", version="0.1.0")

# Include routers
app.include_router(upload.router, prefix="", tags=["upload"])
app.include_router(qa.router, prefix="", tags=["qa"])
app.include_router(summary.router, prefix="", tags=["summary"])