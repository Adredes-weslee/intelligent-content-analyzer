"""Entry point for the LLM generation microservice.

This service is responsible for composing answers to user questions using
retrieved context. In a real deployment this would call into a large
language model API (e.g. OpenAI, Google Gemini) with carefully crafted
prompts and strict citation rules. Here we simply concatenate the
contents of the retrieved chunks to form a dummy answer and attach
citations.
"""

from __future__ import annotations

from typing import List
from fastapi import FastAPI

from shared.models import QARequest, QAResponse, Citation


app = FastAPI(title="LLM Generation Service", version="0.1.0")


@app.post("/generate", response_model=QAResponse)
async def generate_answer(request: QARequest) -> QAResponse:
    """Produce a stub answer for a given question.

    In the absence of an actual language model this handler simply
    echoes back the question and returns no citations. The confidence
    score is always zero.
    """
    # In a real system you would call the retrieval service here to
    # obtain context, then invoke an LLM. For now we return a trivial
    # response.
    answer = f"I'm sorry, I don't have a real answer for: {request.question}"
    citations: List[Citation] = []
    return QAResponse(answer=answer, citations=citations, confidence=0.0, diagnostics={})