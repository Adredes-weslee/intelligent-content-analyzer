import os

# Force offline (no OCR/LLM calls)
os.environ["OFFLINE_MODE"] = "1"

from services.ingest.app.readers import (  # type: ignore
    _parse_csv,
    _parse_html,
    _parse_image,
    _table_to_csv,
    parse_document,
)


def test_parse_document_text_fallback_utf8() -> None:
    b = "Hello world — UTF8.".encode("utf-8")
    out = parse_document(b, "note.txt")
    assert "Hello world" in out


def test_parse_document_markdown_passthrough() -> None:
    md = b"# Title\n\nSome content."
    out = parse_document(md, "readme.md")
    assert "# Title" in out and "Some content" in out


def test_parse_html_basic_extraction() -> None:
    html = b"<html><body><h1>Header</h1><p>Para</p><script>x()</script></body></html>"
    out = _parse_html(html)
    assert "Header" in out and "Para" in out
    assert "x()" not in out  # script removed


def test_parse_csv_prefers_csv_over_excel() -> None:
    csv = b"col1,col2\n1,2\n3,4\n"
    out = _parse_csv(csv)
    assert "col1,col2" in out
    assert "1,2" in out and "3,4" in out


def test_parse_image_offline_returns_empty() -> None:
    # Random bytes that won't decode to an image will return empty string
    out = _parse_image(b"\x00\x01\x02\x03")
    assert out == ""


def test_table_to_csv_normalizes_cells() -> None:
    tbl = [["A", "B\nC"], ["", None], ["1", 2]]
    out = _table_to_csv(tbl)
    assert out.splitlines()[0] == "A,B C"
    assert "1,2" in out
