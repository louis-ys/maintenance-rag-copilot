"""페이지 텍스트를 오버랩이 있는 청크로 분할한다."""

from __future__ import annotations

from src.models import Chunk, PageContent

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def chunk_pages(
    pages: list[PageContent],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """페이지 메타데이터를 유지하면서 각 페이지를 독립적으로 청크로 분할한다.

    페이지 경계를 넘어 텍스트가 섞이지 않도록 페이지 단위로만 분할한다.
    """

    chunks: list[Chunk] = []
    counter = 0

    for page in pages:
        for text in _split_text(page.text, chunk_size, overlap):
            chunk = Chunk(
                chunk_id=f"{page.document_name}-p{page.page_number}-{counter}",
                document_name=page.document_name,
                page_number=page.page_number,
                text=text,
            )
            chunks.append(chunk)
            counter += 1

    return chunks


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """단일 페이지 텍스트를 chunk_size 길이, overlap 겹침으로 분할한다."""

    if len(text) <= chunk_size:
        return [text]

    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("overlap은 chunk_size보다 작아야 합니다.")

    segments: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        segment = text[start:end].strip()
        if segment:
            segments.append(segment)
        if end == length:
            break
        start += step

    return segments
