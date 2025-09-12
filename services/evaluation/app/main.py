"""Entry point for the evaluation microservice.

This service computes basic quality metrics for answers produced by
the system. It provides a simple /evaluate endpoint that takes a
question, an answer and the supporting sources and outputs scores for
factuality, relevance and completeness. Real implementations could use
LLM‑based grading or external libraries such as ragas.
"""

from __future__ import annotations

from fastapi import FastAPI
from shared.models import EvaluateRequest, EvaluateResponse

from .metrics import simple_metric_scores


app = FastAPI(title="Evaluation Service", version="0.1.0")


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    """Compute quality metrics for a generated answer.

    The metrics are heuristic and rely on string containment. They are
    intended solely for demonstration and do not reflect the
    sophisticated evaluation required for real RAG systems.
    """
    factuality, relevance, completeness = simple_metric_scores(
        req.question, req.answer, req.sources
    )
    return EvaluateResponse(
        factuality=factuality,
        relevance=relevance,
        completeness=completeness,
        comments=None,
    )