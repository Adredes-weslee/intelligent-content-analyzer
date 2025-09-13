import os

from fastapi.testclient import TestClient

os.environ["OFFLINE_MODE"] = "1"

from services.ingest.app.main import app  # type: ignore


def test_ingest_basic_text_file() -> None:
    client = TestClient(app)
    content = b"Hello world. This is a test document."
    resp = client.post(
        "/ingest",
        files={"file": ("test.txt", content, "text/plain")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "doc_id" in data
    assert "chunks" in data
    assert isinstance(data["chunks"], list)
    assert len(data["chunks"]) >= 1
