"""End‑to‑end tests for the API gateway."""

from fastapi.testclient import TestClient

from services.api_gateway.app.main import app


def test_api_gateway_workflow() -> None:
    client = TestClient(app)
    # Upload a document
    content = b"FastAPI makes it easy to build APIs."
    upload_resp = client.post(
        "/upload_document",
        files={"file": ("guide.txt", content, "text/plain")},
    )
    assert upload_resp.status_code == 200
    doc_data = upload_resp.json()
    doc_id = doc_data["doc_id"]
    # Ask a question
    qa_resp = client.post(
        "/ask_question",
        json={"question": "What does FastAPI make easy?", "k": 3, "use_rerank": False},
    )
    assert qa_resp.status_code == 200
    qa_data = qa_resp.json()
    assert "answer" in qa_data
    assert "citations" in qa_data
    assert 0.0 <= qa_data["confidence"] <= 1.0
    # Request a summary
    summary_resp = client.get(f"/document_summary?doc_id={doc_id}")
    assert summary_resp.status_code == 200
    summary_data = summary_resp.json()
    assert summary_data["doc_id"] == doc_id
    assert "summary" in summary_data