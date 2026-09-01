"""src.pdf_loader 단위 테스트."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pdf_loader import clean_text, load_pdf_pages

REAL_PDF_PATH = Path("data/manuals/hyosung_motor_manual_ko.pdf")
DOCUMENT_NAME = "hyosung_motor_manual_ko.pdf"


def test_clean_text_collapses_whitespace_and_newlines() -> None:
    raw = "모터   진동\n\n소음   발생\t확인"
    assert clean_text(raw) == "모터 진동 소음 발생 확인"


def test_clean_text_strips_leading_trailing_whitespace() -> None:
    assert clean_text("  \n  텍스트  \n  ") == "텍스트"


def test_clean_text_removes_toc_dot_leaders() -> None:
    raw = "1.1 서론 ..................................................... 3"
    assert clean_text(raw) == "1.1 서론 3"


def test_load_pdf_pages_raises_korean_error_for_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "없는파일.pdf"

    with pytest.raises(FileNotFoundError) as exc_info:
        load_pdf_pages(missing_path, DOCUMENT_NAME)

    message = str(exc_info.value)
    assert "PDF 파일을 찾을 수 없습니다" in message
    assert str(missing_path.resolve()) in message


@pytest.mark.skipif(not REAL_PDF_PATH.exists(), reason="실제 매뉴얼 PDF가 없습니다.")
def test_load_pdf_pages_returns_valid_pages_from_real_manual() -> None:
    pages = load_pdf_pages(REAL_PDF_PATH, DOCUMENT_NAME)

    assert len(pages) > 0

    for page in pages:
        assert page.document_name == DOCUMENT_NAME
        assert page.page_number >= 1
        assert page.text.strip() != ""
        assert "\n" not in page.text

    page_numbers = [page.page_number for page in pages]
    assert page_numbers == sorted(page_numbers)
    assert len(page_numbers) == len(set(page_numbers))
    assert pages[0].page_number >= 1
