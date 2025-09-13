import os

# Offline deterministic embeddings
os.environ["OFFLINE_MODE"] = "1"

from services.embeddings.app.embeddings import embed_texts  # type: ignore
from shared.settings import Settings


def test_embed_texts_single_and_batch() -> None:
    s = Settings()
    v1 = embed_texts(["hello world"])[0]
    v2 = embed_texts(["hello world"])[0]
    assert isinstance(v1, list)
    assert len(v1) == s.embedding_dim
    # Deterministic offline: same input -> same vector
    assert v1 == v2

    batch = embed_texts(["alpha", "beta", "gamma"])
    assert len(batch) == 3
    assert all(len(v) == s.embedding_dim for v in batch)


def test_embed_texts_empty_input_returns_empty() -> None:
    assert embed_texts([]) == []
