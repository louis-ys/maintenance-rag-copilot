"""RAG 핵심 검색 기능 스모크 테스트.

data/manuals/hyosung_motor_manual_ko.pdf 를 색인한 뒤 지정된 4개 질문을
검색하여 결과를 지정된 형식으로 출력한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.models import SearchResponse
from src.rag_agent import RagAgent

QUESTIONS = [
    "모터에서 심한 진동과 금속성 소음이 발생합니다.",
    "베어링 그리스는 어떻게 보충해야 하나요?",
    "모터가 과열될 때 무엇을 확인해야 하나요?",
    "오늘 서울 날씨는 어떤가요?",
]


def print_response(response: SearchResponse) -> None:
    print("=" * 50)
    print(f"질문: {response.query}")
    evidence_label = "부족 (insufficient_evidence=True)" if response.insufficient_evidence else "충분"
    print(f"근거 충분 여부: {evidence_label}")

    if not response.results:
        print("검색 결과 없음")
    for result in response.results:
        print(f"검색 결과 {result.rank}:")
        print(f"- 문서명: {result.document_name}")
        print(f"- PDF 페이지: {result.page_number}")
        print(f"- 유사도: {result.score:.4f}")
        print(f"- 원문: {result.text}")
    print("=" * 50)
    print()


def main() -> None:
    agent = RagAgent()

    print("PDF 색인을 생성(또는 캐시 로드)하는 중입니다...")
    agent.build_index()
    print(f"유효 페이지 수: {agent.page_count}")
    print(f"생성된 청크 수: {agent.chunk_count}")
    print()

    for question in QUESTIONS:
        response = agent.search(question)
        print_response(response)


if __name__ == "__main__":
    main()
