import os

from fastapi.testclient import TestClient

# Force offline to avoid network
os.environ["OFFLINE_MODE"] = "1"

import os

from services.llm_generate.app.main import (  # type: ignore
    _map_bracket_citations,
    _route_model,
    app,
)
from shared.models import DocChunk, DocMetadata


def test_generate_offline_with_context() -> None:
    client = TestClient(app)
    ctx = [
        DocChunk(
            id="c1",
            doc_id="d1",
            text="FastAPI makes it easy to build APIs.",
            meta=DocMetadata(source="doc1", page=1),
        ).dict()
    ]
    resp = client.post(
        "/generate",
        json={"question": "What does FastAPI make easy?", "context_chunks": ctx},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data and isinstance(data["answer"], str)
    assert isinstance(data.get("citations"), list)


def test_summarize_offline() -> None:
    client = TestClient(app)
    ctx = [
        DocChunk(
            id="c1",
            doc_id="d1",
            text="Section 1: Intro to FastAPI.",
            meta=DocMetadata(source="doc1", page=1),
        ).dict()
    ]
    resp = client.post("/summarize", json={"doc_id": "d1", "chunks": ctx})
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert data.get("doc_id") == "d1"
    assert isinstance(data.get("key_points"), list)
    assert isinstance(data.get("citations"), list)


def test_route_model_fields_present() -> None:
    route = _route_model("Explain HTTP GET vs POST")
    assert isinstance(route, dict)
    assert route.get("tier") in {"fast", "reasoning"}
    assert isinstance(route.get("model"), str)
    assert isinstance(route.get("why"), str)


def test_generate_offline_without_context() -> None:
    client = TestClient(app)
    resp = client.post("/generate", json={"question": "What is FastAPI?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert isinstance(data.get("citations"), list)
    assert (
        "insufficient" in data["answer"].lower() or "fallback" in data["answer"].lower()
    )


def test_map_bracket_citations_parses_indices() -> None:
    txt = (
        "See details in [1] and [2], duplicate [2] and ignore [12] out of range later."
    )
    idxs = _map_bracket_citations(txt)
    assert idxs == [0, 1, 11]  # zero-based, de-duped in order


def test_summarize_offline_no_chunks() -> None:
    client = TestClient(app)
    resp = client.post("/summarize", json={"doc_id": "empty", "chunks": []})
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("doc_id") == "empty"
    assert data["summary"].lower().startswith("no content")
    assert data.get("citations") == []
