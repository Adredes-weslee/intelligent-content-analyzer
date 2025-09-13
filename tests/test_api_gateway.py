import os
from typing import List
from unittest.mock import patch

from fastapi.testclient import TestClient

from services.api_gateway.app.routers import qa as qa_router

# Force offline and disable judge to avoid external calls
os.environ["OFFLINE_MODE"] = "1"
os.environ["EVAL_LLM_ENABLED"] = "0"

import os

from services.api_gateway.app.main import app
from shared.cache import (
    bump_index_version,
    content_fingerprint,
    get_index_version,
    semantic_key,
)
from shared.tracing import log_event, span

# Isolate gateway; avoid indexing side-effects and OCR
os.environ["OFFLINE_MODE"] = "1"

from shared.models import (
    Citation,
    DocChunk,
    DocMetadata,
    EvaluateRequest,
    EvaluateResponse,
    QARequest,
    QAResponse,
    RetrieveRequest,
    RetrieveResponse,
    RetrieveResult,
)

client = TestClient(app)


def _stub_hits() -> List[RetrieveResult]:
    meta = DocMetadata(source="doc1", page=1, section="Intro")
    chunk = DocChunk(id="c1", doc_id="d1", text="FastAPI makes APIs easy.", meta=meta)
    return [RetrieveResult(chunk=chunk, score=0.9, bm25=1.0, dense=0.8)]


def _stub_retrieval(req: RetrieveRequest) -> RetrieveResponse:
    return RetrieveResponse(hits=_stub_hits(), diagnostics={"mode": "stub"})


def _stub_generate(req: QARequest) -> QAResponse:
    c = _stub_hits()[0].chunk
    return QAResponse(
        answer="FastAPI makes building APIs easy.",
        citations=[Citation(doc_id=c.doc_id, page=c.meta.page, section=c.meta.section)],
        confidence=0.0,
        diagnostics={"mode": "stub"},
    )


async def _stub_evaluate(req: EvaluateRequest) -> EvaluateResponse:
    return EvaluateResponse(
        factuality=1.0,
        relevance=1.0,
        completeness=1.0,
        faithfulness=1.0,
        answer_relevance_1_5=5.0,
        context_relevance_ratio=1.0,
        comments="stub",
    )


def test_ask_question_unit_isolated() -> None:
    with (
        patch(
            "services.api_gateway.app.routers.qa.retrieval_search",
            side_effect=_stub_retrieval,
        ),
        patch(
            "services.api_gateway.app.routers.qa.llm_generate",
            side_effect=_stub_generate,
        ),
        patch("services.evaluation.app.main.evaluate", side_effect=_stub_evaluate),
    ):
        resp = client.post(
            "/ask_question",
            json={
                "question": "What does FastAPI make easy?",
                "k": 3,
                "use_rerank": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data and isinstance(data["answer"], str)
        assert isinstance(data.get("citations"), list)
        assert 0.0 <= data.get("confidence", 0.0) <= 1.0


def test_upload_document_unit_isolated() -> None:
    fake_text = "Hello world. Unit testing upload."
    # Patch heavy dependencies inside upload flow
    with (
        patch(
            "services.api_gateway.app.routers.upload.parse_document",
            return_value=fake_text,
        ),
        patch(
            "services.api_gateway.app.routers.upload.iter_section_chunks",
            return_value=iter([(fake_text, 1, "S", None)]),
        ),
        patch("services.api_gateway.app.routers.upload.add_chunks", return_value=None),
        patch(
            "services.api_gateway.app.routers.upload.lookup_by_checksum",
            return_value=None,
        ),
        patch("services.api_gateway.app.routers.upload.lookup_file", return_value=None),
        patch("services.api_gateway.app.routers.upload.track_doc", return_value=None),
        patch(
            "services.api_gateway.app.routers.upload.update_file_entry",
            return_value=None,
        ),
        patch(
            "services.api_gateway.app.routers.upload.bump_index_version", return_value=1
        ),
    ):
        resp = client.post(
            "/upload_document",
            files={"file": ("unit.txt", b"Hello world", "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "doc_id" in data
        assert "num_chunks" in data


class DummyCache:
    def __init__(self):
        self.store = {}

    def get(self, k):
        return self.store.get(k)

    def set(self, k, v, ex=None):
        self.store[k] = v


def test_content_fingerprint_determinism() -> None:
    payload = {"a": 1, "b": ["x", "y"], "c": {"k": "v"}}
    fp1 = content_fingerprint(payload)
    fp2 = content_fingerprint(payload)
    assert fp1 == fp2
    assert isinstance(fp1, str) and len(fp1) >= 8


def test_semantic_key_generation() -> None:
    key = semantic_key("Hello world")
    assert isinstance(key, str)
    # Validate "prefix:hexdigest" shape without assuming exact prefix text
    assert ":" in key
    prefix, digest = key.split(":", 1)
    assert prefix  # non-empty prefix
    hexchars = set("0123456789abcdef")
    s = digest.lower()
    assert len(s) in (40, 64) and all(c in hexchars for c in s)


def test_index_version_bump_with_dummy_cache(monkeypatch) -> None:
    dummy = DummyCache()
    monkeypatch.setattr("shared.cache.get_default_cache", lambda: dummy, raising=True)
    v0 = get_index_version()
    v1 = bump_index_version()
    v2 = bump_index_version()
    assert isinstance(v0, int)
    assert v1 == v0 + 1
    assert v2 == v1 + 1


def test_tracing_span_noop_offline() -> None:
    with span("unit.test", foo=1, bar="x"):
        log_event("inside-span", payload={"k": "v"})
    assert True


class _CountingCache:
    def __init__(self):
        self.store = {}
        self.set_calls = 0

    def get(self, k):
        return self.store.get(k)

    # qa.py uses _cache.set(..., ttl=...)
    def set(self, k, v, ttl=None, **kwargs):
        self.set_calls += 1
        self.store[k] = v


def test_ask_question_writes_to_cache(monkeypatch) -> None:
    # Ensure caching paths are enabled in the router
    if hasattr(qa_router, "_settings"):
        monkeypatch.setattr(
            qa_router._settings, "rate_limit_enabled", True, raising=True
        )
        monkeypatch.setattr(qa_router._settings, "cache_enabled", True, raising=True)

    c = _CountingCache()
    monkeypatch.setattr("services.api_gateway.app.routers.qa._cache", c, raising=True)

    with (
        patch(
            "services.api_gateway.app.routers.qa.retrieval_search",
            side_effect=_stub_retrieval,
        ),
        patch(
            "services.api_gateway.app.routers.qa.llm_generate",
            side_effect=_stub_generate,
        ),
        patch("services.evaluation.app.main.evaluate", side_effect=_stub_evaluate),
    ):
        resp = client.post(
            "/ask_question", json={"question": "cache me", "k": 2, "use_rerank": False}
        )
        assert resp.status_code == 200
        # At least the rate-limit write should fire; response writes also run when cache_enabled=True
        assert c.set_calls >= 1
