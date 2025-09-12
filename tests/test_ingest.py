"""Tests for the ingest service."""

from fastapi.testclient import TestClient

from services.ingest.app.main import app


def test_ingest_returns_chunks() -> None:
    client = TestClient(app)
    # Upload a small text file
    content = b"Hello world. This is a test document."
    response = client.post(
        "/ingest",
        files={"file": ("test.txt", content, "text/plain")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "doc_id" in data
    assert "chunks" in data
    # At least one chunk should be returned
    assert len(data["chunks"]) >= 1