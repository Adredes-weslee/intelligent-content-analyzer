import os

from fastapi.testclient import TestClient

os.environ["OFFLINE_MODE"] = "1"
os.environ["EVAL_LLM_ENABLED"] = "0"

import os

from services.evaluation.app.main import app as eval_app  # type: ignore
from shared.models import DocChunk, DocMetadata


def test_evaluate_heuristics_only() -> None:
    client = TestClient(eval_app)
    sources = [
        DocChunk(
            id="c1",
            doc_id="d1",
            text="The sky is blue.",
            meta=DocMetadata(source="doc1"),
        ).dict(),
        DocChunk(
            id="c2",
            doc_id="d1",
            text="Grass is green.",
            meta=DocMetadata(source="doc1"),
        ).dict(),
    ]
    resp = client.post(
        "/evaluate",
        json={
            "question": "What color is the sky?",
            "answer": "The sky is blue.",
            "sources": sources,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["factuality"] <= 1.0
    assert 0.0 <= data["relevance"] <= 1.0
    assert 0.0 <= data["completeness"] <= 1.0


def test_judge_endpoint_offline_returns_heuristics() -> None:
    client = TestClient(eval_app)
    sources = [
        DocChunk(
            id="c1",
            doc_id="d1",
            text="Paris is the capital of France.",
            meta=DocMetadata(source="doc"),
        ).model_dump()
    ]
    resp = client.post(
        "/judge",
        json={
            "question": "What is the capital of France?",
            "answer": "Paris is the capital of France.",
            "sources": sources,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "factuality" in data
    assert "relevance" in data
    assert "completeness" in data


def test_evaluate_ctx_ratio_none_without_hits() -> None:
    client = TestClient(eval_app)
    sources = [
        DocChunk(
            id="c1",
            doc_id="d1",
            text="FastAPI is a Python framework.",
            meta=DocMetadata(source="doc"),
        ).dict()
    ]
    resp = client.post(
        "/evaluate",
        json={
            "question": "What is FastAPI?",
            "answer": "A Python framework.",
            "sources": sources,
            "hits": [],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert (
        data.get("context_relevance_ratio") in (None, 0.0)
        or "context_relevance_ratio" not in data
    )


def test_evaluate_ctx_ratio_with_hits() -> None:
    client = TestClient(eval_app)
    chunk = DocChunk(
        id="c1",
        doc_id="d1",
        text="FastAPI is a Python web framework.",
        meta=DocMetadata(source="doc"),
    ).dict()
    hit = {
        "chunk": chunk,
        "score": 1.0,
        "bm25": 1.0,
        "dense": 0.0,
    }
    resp = client.post(
        "/evaluate",
        json={
            "question": "What is FastAPI?",
            "answer": "A Python framework.",
            "sources": [chunk],
            "hits": [hit],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    val = data.get("context_relevance_ratio")
    assert val is None or (0.0 <= float(val) <= 1.0)
