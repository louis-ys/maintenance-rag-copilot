"""문장 임베딩 기반 벡터 검색 인덱스와 디스크 캐시."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.keyword_rules import compute_keyword_bonus
from src.models import Chunk, SearchResult

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_SIMILARITY_THRESHOLD = 0.25
DEFAULT_TOP_K = 3

# 텍스트 정제, 청킹, 임베딩, 점수 계산 로직이 바뀔 때마다 값을 올려서
# 기존 디스크 캐시(data/index/)가 자동으로 무효화되도록 한다.
INDEX_VERSION = "2-hybrid-keyword-bonus"

_EMBEDDINGS_FILE = "embeddings.npy"
_META_FILE = "meta.json"


def compute_source_signature(pdf_path: Path, chunk_size: int, overlap: int) -> str:
    """PDF 파일 상태와 청킹 설정을 반영한 캐시 무효화용 서명값을 만든다."""

    stat = pdf_path.stat()
    payload = f"{pdf_path.name}:{stat.st_size}:{stat.st_mtime_ns}:{chunk_size}:{overlap}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class EmbeddingIndex:
    """청크 임베딩을 생성/캐시하고 코사인 유사도 검색을 수행한다."""

    def __init__(
        self,
        index_dir: Path,
        model_name: str = EMBEDDING_MODEL_NAME,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self.index_dir = index_dir
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self._model: SentenceTransformer | None = None
        self.embeddings: np.ndarray | None = None
        self.chunks: list[Chunk] = []

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build_or_load(self, chunks: list[Chunk], source_signature: str) -> None:
        """캐시가 유효하면 로드하고, 아니면 새로 임베딩을 생성해 저장한다."""

        if self._try_load_cache(source_signature):
            return
        self._build_embeddings(chunks)
        self._save_cache(source_signature)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> tuple[list[SearchResult], bool]:
        """질의에 대해 상위 top_k개의 하이브리드 검색 결과와 근거 부족 여부를 반환한다.

        의미 검색(코사인 유사도) 점수에 한국어 키워드 확장 가산점을 더한
        final_score로 전체 청크를 재정렬한다.
        """

        if self.embeddings is None or not self.chunks:
            raise RuntimeError("인덱스가 아직 생성되지 않았습니다. build_or_load를 먼저 호출하세요.")

        query_embedding = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        semantic_scores = self.embeddings @ query_embedding

        scored_indices = self._score_all_chunks(query, semantic_scores)
        top_items = scored_indices[:top_k]

        results = [
            SearchResult(
                rank=rank + 1,
                document_name=self.chunks[idx].document_name,
                page_number=self.chunks[idx].page_number,
                text=self.chunks[idx].text,
                semantic_score=semantic_score,
                keyword_bonus=keyword_bonus,
                final_score=final_score,
                score=final_score,
            )
            for rank, (idx, semantic_score, keyword_bonus, final_score) in enumerate(top_items)
        ]

        insufficient_evidence = not results or results[0].final_score < self.similarity_threshold
        return results, insufficient_evidence

    def _score_all_chunks(
        self, query: str, semantic_scores: np.ndarray
    ) -> list[tuple[int, float, float, float]]:
        """전체 청크에 대해 (인덱스, 의미점수, 키워드보너스, 최종점수)를 계산해 정렬한다."""

        scored: list[tuple[int, float, float, float]] = []
        for idx, chunk in enumerate(self.chunks):
            semantic_score = float(semantic_scores[idx])
            keyword_bonus = compute_keyword_bonus(query, chunk.text)
            final_score = semantic_score + keyword_bonus
            scored.append((idx, semantic_score, keyword_bonus, final_score))

        scored.sort(key=lambda item: item[3], reverse=True)
        return scored

    def _try_load_cache(self, source_signature: str) -> bool:
        meta_path = self.index_dir / _META_FILE
        embeddings_path = self.index_dir / _EMBEDDINGS_FILE
        if not meta_path.exists() or not embeddings_path.exists():
            return False

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("source_signature") != source_signature
            or meta.get("model_name") != self.model_name
            or meta.get("index_version") != INDEX_VERSION
        ):
            return False

        self.embeddings = np.load(embeddings_path)
        self.chunks = [Chunk(**item) for item in meta["chunks"]]
        return True

    def _build_embeddings(self, chunks: list[Chunk]) -> None:
        texts = [chunk.text for chunk in chunks]
        self.embeddings = self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        self.chunks = chunks

    def _save_cache(self, source_signature: str) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.index_dir / _EMBEDDINGS_FILE, self.embeddings)
        meta = {
            "source_signature": source_signature,
            "model_name": self.model_name,
            "index_version": INDEX_VERSION,
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }
        (self.index_dir / _META_FILE).write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
