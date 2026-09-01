"""pypdf 기반 PDF 페이지 로더."""

from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from src.models import PageContent

_WHITESPACE_RE = re.compile(r"\s+")
_DOT_LEADER_RE = re.compile(r"(?:\.\s?){3,}")


def clean_text(text: str) -> str:
    """목차의 점선 leader와 연속된 공백/줄바꿈을 정리한다.

    "..... ..... ....." 같은 목차 점선(leader)은 실제 내용이 아니라 PDF
    레이아웃 장식이며, 그대로 두면 청크 임베딩이 왜곡되어 목차 페이지가
    본문보다 높은 유사도로 검색되는 문제가 발생하므로 제거한다.
    """

    without_dot_leaders = _DOT_LEADER_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", without_dot_leaders).strip()


def load_pdf_pages(pdf_path: Path, document_name: str) -> list[PageContent]:
    """PDF를 페이지 단위로 읽어 텍스트가 있는 페이지만 반환한다.

    page_number는 실제 PDF 페이지 번호이며 첫 페이지가 1이 된다.
    """

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF 파일을 찾을 수 없습니다. 경로를 확인하세요: {pdf_path.resolve()}"
        )

    reader = PdfReader(str(pdf_path))
    pages: list[PageContent] = []

    for index, page in enumerate(reader.pages):
        page_number = index + 1
        raw_text = page.extract_text() or ""
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue
        pages.append(
            PageContent(
                document_name=document_name,
                page_number=page_number,
                text=cleaned,
            )
        )

    return pages
