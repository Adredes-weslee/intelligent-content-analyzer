import asyncio
import os

from services.retrieval.app.main import INDEX
from services.retrieval.app.main import search as bm25_search
from shared.models import DocChunk, DocMetadata, RetrieveRequest

# Keep retrieval isolated and deterministic
os.environ["OFFLINE_MODE"] = "1"

from services.retrieval.app.hybrid import hybrid_search  # type: ignore
from services.retrieval.app.main import INDEX as RETRIEVAL_INDEX  # type: ignore


def make_chunk(cid: str, text: str, doc_id: str = "d1") -> DocChunk:
    return DocChunk(id=cid, doc_id=doc_id, text=text, meta=DocMetadata(source="doc1"))


def test_hybrid_search_unit_with_in_memory_index() -> None:
    # Seed a minimal in-memory index (BM25 will operate over INDEX)
    RETRIEVAL_INDEX.clear()
    docs: List[DocChunk] = [
        make_chunk("c1", "The sky is blue and the sun is bright."),
        make_chunk("c2", "Grass is green and trees are tall."),
        make_chunk("c3", "FastAPI makes building APIs easy."),
    ]
    RETRIEVAL_INDEX.extend(docs)

    req = RetrieveRequest(query="What color is the sky?", top_k=2, hybrid=True)
    res = __import__("asyncio").get_event_loop().run_until_complete(hybrid_search(req))
    assert res.hits is not None
    assert len(res.hits) >= 1
    # Top hit should be relevant to "sky"
    assert any("sky" in h.chunk.text.lower() for h in res.hits[:2])


def _mk_chunk(cid: str, text: str, doc_id: str = "d1") -> DocChunk:
    return DocChunk(id=cid, doc_id=doc_id, text=text, meta=DocMetadata(source="doc1"))


def test_bm25_only_search_over_in_memory_index() -> None:
    INDEX.clear()
    docs = [
        _mk_chunk("c1", "HTTP is a protocol used on the web."),
        _mk_chunk("c2", "POST requests can create resources."),
        _mk_chunk("c3", "GET requests retrieve resources."),
    ]
    INDEX.extend(docs)

    req = RetrieveRequest(query="What do GET requests do?", top_k=2, hybrid=False)
    res = asyncio.get_event_loop().run_until_complete(bm25_search(req))
    assert res.hits is not None and len(res.hits) >= 1
    top_texts = [h.chunk.text.lower() for h in res.hits]
    assert any("get requests" in t or "retrieve" in t for t in top_texts)


def test_bm25_search_empty_index_returns_no_hits() -> None:
    INDEX.clear()
    req = RetrieveRequest(query="anything", top_k=5, hybrid=False)
    res = asyncio.get_event_loop().run_until_complete(bm25_search(req))
    assert res.hits is not None
    assert len(res.hits) == 0


def test_bm25_top_k_does_not_exceed_index_size() -> None:
    INDEX.clear()
    INDEX.append(_mk_chunk("c1", "Only one document present."))
    req = RetrieveRequest(query="document", top_k=10, hybrid=False)
    res = asyncio.get_event_loop().run_until_complete(bm25_search(req))
    assert len(res.hits) <= 1
