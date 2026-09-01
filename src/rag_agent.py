"""PDF 로딩, 청킹, 임베딩 검색을 하나로 묶은 RAG 검색 에이전트."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from src.chunker import CHUNK_OVERLAP, CHUNK_SIZE, chunk_pages
from src.embedding_index import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    EmbeddingIndex,
    compute_source_signature,
)
from src.models import SearchResponse
from src.pdf_loader import load_pdf_pages

DEFAULT_PDF_PATH = Path("data/manuals/hyosung_motor_manual_ko.pdf")
DEFAULT_DOCUMENT_NAME = "hyosung_motor_manual_ko.pdf"
DEFAULT_INDEX_DIR = Path("data/index")


class RagAgent:
    """효성 저압전동기 매뉴얼에 대한 근거 기반 검색만 수행하는 1단계 에이전트."""

    def __init__(
        self,
        pdf_path: Path = DEFAULT_PDF_PATH,
        document_name: str = DEFAULT_DOCUMENT_NAME,
        index_dir: Path = DEFAULT_INDEX_DIR,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self.pdf_path = pdf_path
        self.document_name = document_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index = EmbeddingIndex(
            index_dir=index_dir, similarity_threshold=similarity_threshold
        )
        self.page_count = 0
        self.chunk_count = 0
        self.page_text_by_number: dict[int, str] = {}

    def build_index(self) -> None:
        """PDF를 읽어 청킹하고 임베딩 인덱스를 생성(또는 캐시 로드)한다.

        검색용 청크와는 별도로, 화면 표시(페이지 전체 보기)에 쓸 페이지별
        전체 텍스트를 page_text_by_number에 유지한다. 이 값은 검색/임베딩
        인덱스에는 전혀 사용되지 않는다.
        """

        pages = load_pdf_pages(self.pdf_path, self.document_name)
        self.page_count = len(pages)
        self.page_text_by_number = {page.page_number: page.text for page in pages}

        chunks = chunk_pages(pages, self.chunk_size, self.chunk_overlap)
        self.chunk_count = len(chunks)

        signature = compute_source_signature(self.pdf_path, self.chunk_size, self.chunk_overlap)
        self.index.build_or_load(chunks, signature)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> SearchResponse:
        """질의에 대한 상위 top_k 검색 결과를 반환한다.

        검색 순위와 점수는 청크 기반 검색 결과를 그대로 유지하며,
        각 결과에 해당 page_number의 페이지 전체 텍스트(page_text)만 덧붙인다.
        """

        results, insufficient_evidence = self.index.search(query, top_k=top_k)
        results_with_page_text = [
            replace(
                result,
                page_text=self.page_text_by_number.get(result.page_number, result.text),
            )
            for result in results
        ]
        return SearchResponse(
            query=query,
            insufficient_evidence=insufficient_evidence,
            results=results_with_page_text,
        )
