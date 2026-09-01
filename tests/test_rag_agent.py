"""src.rag_agent 단위/통합 테스트.

임베딩 모델 호출 없이 페이지 전체 텍스트 배선(wiring)을 검증하기 위해
EmbeddingIndex 대신 간단한 스텁을 주입한다. 실제 PDF에 의존하는 테스트는
data/manuals/hyosung_motor_manual_ko.pdf가 있을 때만 실행된다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.chunker import CHUNK_SIZE
from src.models import SearchResult
from src.pdf_loader import load_pdf_pages
from src.rag_agent import RagAgent
from src.source_text_formatter import format_page_text_for_display

REAL_PDF_PATH = Path("data/manuals/hyosung_motor_manual_ko.pdf")
DOCUMENT_NAME = "hyosung_motor_manual_ko.pdf"


class _StubIndex:
    """EmbeddingIndex 대역. 검색 순위/점수 로직은 건드리지 않고 반환값만 통제한다."""

    def __init__(self, results: list[SearchResult], insufficient_evidence: bool) -> None:
        self._results = results
        self._insufficient_evidence = insufficient_evidence

    def search(self, query: str, top_k: int = 3):
        return self._results, self._insufficient_evidence


def _make_chunk_result(page_number: int, text: str) -> SearchResult:
    return SearchResult(
        rank=1,
        document_name=DOCUMENT_NAME,
        page_number=page_number,
        text=text,
        semantic_score=0.9,
        keyword_bonus=0.0,
        final_score=0.9,
        score=0.9,
    )


def test_search_attaches_page_text_longer_than_chunk() -> None:
    agent = RagAgent()
    long_page_text = "그리스 보급과 소음 관련 안내 문단입니다. " * 40
    agent.page_text_by_number = {7: long_page_text}
    chunk_text = "검색된 짧은 청크 텍스트"
    agent.index = _StubIndex([_make_chunk_result(7, chunk_text)], insufficient_evidence=False)

    response = agent.search("그리스 보충 방법")

    result = response.results[0]
    assert result.page_text == long_page_text
    assert len(result.page_text) > len(result.text)
    assert len(result.page_text) > CHUNK_SIZE
    # 검색 순위/점수는 스텁이 반환한 값 그대로 유지되어야 한다.
    assert result.text == chunk_text
    assert result.final_score == 0.9


def test_search_falls_back_to_chunk_text_when_page_missing() -> None:
    agent = RagAgent()
    agent.page_text_by_number = {}
    chunk_text = "청크 텍스트"
    agent.index = _StubIndex([_make_chunk_result(3, chunk_text)], insufficient_evidence=False)

    response = agent.search("질문")

    assert response.results[0].page_text == chunk_text


@pytest.mark.skipif(not REAL_PDF_PATH.exists(), reason="실제 매뉴얼 PDF가 없습니다.")
def test_page_7_display_does_not_start_with_broken_fragment() -> None:
    pages = load_pdf_pages(REAL_PDF_PATH, DOCUMENT_NAME)
    page_7 = next(page for page in pages if page.page_number == 7)

    formatted = format_page_text_for_display(page_7.text)

    assert not formatted.startswith("회 주입량은")


@pytest.mark.skipif(not REAL_PDF_PATH.exists(), reason="실제 매뉴얼 PDF가 없습니다.")
def test_page_7_display_does_not_end_with_incomplete_word() -> None:
    pages = load_pdf_pages(REAL_PDF_PATH, DOCUMENT_NAME)
    page_7 = next(page for page in pages if page.page_number == 7)

    formatted = format_page_text_for_display(page_7.text)

    assert not formatted.rstrip().endswith("과")
