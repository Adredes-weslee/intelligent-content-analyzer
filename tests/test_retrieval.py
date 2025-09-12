"""Tests for the retrieval service."""

from fastapi.testclient import TestClient

from services.retrieval.app.main import app, INDEX
from shared.models import DocChunk, DocMetadata


def test_retrieval_search_basic() -> None:
    client = TestClient(app)
    # Clear any existing index
    INDEX.clear()
    # Index a couple of chunks
    chunk1 = DocChunk(id="d1_c1", doc_id="d1", text="Python is a programming language", meta=DocMetadata(source="test.txt"))
    chunk2 = DocChunk(id="d2_c1", doc_id="d2", text="Cats are lovely animals", meta=DocMetadata(source="test2.txt"))
    resp = client.post("/index", json=[chunk1.dict(), chunk2.dict()])
    assert resp.status_code == 200
    # Search for a term that appears in chunk1
    search_resp = client.post("/search", json={"query": "Python programming", "top_k": 2, "hybrid": True})
    assert search_resp.status_code == 200
    hits = search_resp.json()["hits"]
    assert len(hits) >= 1
    # The first hit should correspond to chunk1
    top_hit = hits[0]
    assert top_hit["chunk"]["doc_id"] == "d1"