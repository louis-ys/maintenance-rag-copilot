"""src.chunker 단위 테스트."""

from __future__ import annotations

from src.chunker import CHUNK_OVERLAP, CHUNK_SIZE, chunk_pages
from src.models import PageContent


def test_short_page_produces_single_chunk() -> None:
    page = PageContent(document_name="doc.pdf", page_number=1, text="짧은 텍스트")

    chunks = chunk_pages([page])

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == "짧은 텍스트"
    assert chunk.document_name == "doc.pdf"
    assert chunk.page_number == 1
    assert chunk.chunk_id != ""


def test_long_page_is_split_with_size_and_overlap() -> None:
    long_text = "가" * 1500
    page = PageContent(document_name="doc.pdf", page_number=3, text=long_text)

    chunks = chunk_pages([page], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= CHUNK_SIZE
        assert chunk.page_number == 3
        assert chunk.document_name == "doc.pdf"

    reconstructed_start = chunks[0].text[: CHUNK_SIZE - CHUNK_OVERLAP]
    assert long_text.startswith(reconstructed_start)


def test_chunks_do_not_cross_page_boundaries() -> None:
    page_one = PageContent(document_name="doc.pdf", page_number=1, text="첫 페이지 " * 50)
    page_two = PageContent(document_name="doc.pdf", page_number=2, text="둘째 페이지 " * 50)

    chunks = chunk_pages([page_one, page_two])

    page_one_chunks = [c for c in chunks if c.page_number == 1]
    page_two_chunks = [c for c in chunks if c.page_number == 2]

    assert len(page_one_chunks) > 0
    assert len(page_two_chunks) > 0
    for chunk in page_one_chunks:
        assert "둘째 페이지" not in chunk.text
    for chunk in page_two_chunks:
        assert "첫 페이지" not in chunk.text


def test_chunk_has_all_required_fields() -> None:
    page = PageContent(document_name="doc.pdf", page_number=5, text="필드 확인용 텍스트")

    chunks = chunk_pages([page])
    chunk = chunks[0]

    assert hasattr(chunk, "chunk_id")
    assert hasattr(chunk, "document_name")
    assert hasattr(chunk, "page_number")
    assert hasattr(chunk, "text")


def test_empty_page_list_produces_no_chunks() -> None:
    assert chunk_pages([]) == []
