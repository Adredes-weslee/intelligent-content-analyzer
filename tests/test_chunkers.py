import os

os.environ["OFFLINE_MODE"] = "1"

from services.ingest.app.chunkers import (  # type: ignore
    _is_heading,
    chunk_text,
    iter_section_chunks,
)


def test_chunk_text_word_boundary() -> None:
    text = " ".join([f"w{i}" for i in range(25)])
    chunks = chunk_text(text, max_tokens=10)
    assert len(chunks) == 3
    assert all(len(c.split()) <= 10 for c in chunks)


def test_is_heading_heuristics() -> None:
    assert _is_heading("# Intro") == "Intro"
    assert _is_heading("1. Overview") is not None
    assert _is_heading("THIS IS A TITLE") == "THIS IS A TITLE"
    assert _is_heading("=== Page 1 ===") is None
    assert _is_heading("[table id=p1_t1]") is None


def test_iter_section_chunks_pages_tables_and_headings() -> None:
    text = """
=== Page 1 ===
# Intro
This is the first section with some text.
[table id=p1_t1]
a,b,c
d,e,f
[/table]

=== Page 2 ===
2. Methods
More text goes here continuing the discussion.
"""
    out = list(
        iter_section_chunks(
            text, max_tokens=6, respect_pages=True, respect_headings=True
        )
    )
    assert len(out) >= 3
    for chunk, page, section, table_id in out:
        assert isinstance(chunk, str)
        assert page in {1, 2}
        if table_id is not None:
            assert section in {"Intro", "2. Methods"}
            assert table_id == "p1_t1"
        else:
            assert table_id is None
