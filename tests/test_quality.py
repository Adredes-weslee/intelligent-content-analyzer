"""Tests for the evaluation and confidence modules."""

from fastapi.testclient import TestClient

from services.evaluation.app.main import app as eval_app
from services.evaluation.app.confidence import compute_confidence
from shared.models import DocChunk, DocMetadata, RetrieveResult


def test_evaluation_scores() -> None:
    client = TestClient(eval_app)
    # Define a simple request
    sources = [
        DocChunk(id="c1", doc_id="d1", text="The sky is blue", meta=DocMetadata(source="doc1")),
        DocChunk(id="c2", doc_id="d1", text="Grass is green", meta=DocMetadata(source="doc1")),
    ]
    resp = client.post(
        "/evaluate",
        json={
            "question": "What color is the sky?",
            "answer": "The sky is blue",
            "sources": [chunk.dict() for chunk in sources],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["factuality"] <= 1.0
    assert 0.0 <= data["relevance"] <= 1.0
    assert 0.0 <= data["completeness"] <= 1.0


def test_confidence_calculation() -> None:
    # Create dummy retrieval hits
    chunk = DocChunk(id="c1", doc_id="d1", text="Python is great", meta=DocMetadata(source="doc"))
    hit = RetrieveResult(chunk=chunk, score=0.8, bm25=0.8, dense=0.8)
    eval_scores = {"factuality": 0.9, "relevance": 0.9, "completeness": 1.0}
    confidence = compute_confidence([hit], eval_scores)
    assert 0.0 <= confidence <= 1.0