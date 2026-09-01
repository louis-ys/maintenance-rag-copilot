"""RAG 파이프라인에서 사용하는 공용 데이터 모델."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageContent:
    """PDF에서 추출한 한 페이지의 정제된 텍스트."""

    document_name: str
    page_number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    """페이지 메타데이터를 유지한 채 분할된 텍스트 조각."""

    chunk_id: str
    document_name: str
    page_number: int
    text: str


@dataclass(frozen=True)
class SearchResult:
    """검색 결과 한 건.

    semantic_score는 임베딩 코사인 유사도, keyword_bonus는 키워드 확장
    규칙에 따른 가산점, final_score는 둘의 합이며 정렬 기준이 된다.
    score는 final_score와 동일한 값으로, 기존 코드와의 호환을 위해 유지한다.
    page_text는 검색/임베딩에는 쓰이지 않는 화면 표시 전용 필드로,
    text(검색 청크)가 속한 PDF 페이지의 전체 텍스트를 담는다.
    """

    rank: int
    document_name: str
    page_number: int
    text: str
    semantic_score: float
    keyword_bonus: float
    final_score: float
    score: float
    page_text: str = ""


@dataclass
class SearchResponse:
    """질의 하나에 대한 검색 응답 전체."""

    query: str
    insufficient_evidence: bool
    results: list[SearchResult]
