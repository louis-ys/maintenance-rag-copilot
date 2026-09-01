"""src.orchestrator 단위 테스트.

실제 임베딩 모델을 호출하지 않도록, RagAgent와 동일한 .search() 인터페이스를
가진 스텁(stub)을 주입해 Supervisor -> RAG -> Maintenance 흐름만 검증한다.
"""

from __future__ import annotations

from src.models import SearchResponse, SearchResult
from src.orchestrator import run_pipeline

DOCUMENT_NAME = "hyosung_motor_manual_ko.pdf"


class StubRagAgent:
    """테스트용 RagAgent 대역. search() 호출 여부와 반환값을 통제한다."""

    def __init__(self, response: SearchResponse) -> None:
        self._response = response
        self.search_called = False
        self.last_query: str | None = None

    def search(self, query: str, top_k: int = 3) -> SearchResponse:
        self.search_called = True
        self.last_query = query
        return self._response


def _make_result(page_number: int, text: str, score: float = 0.6) -> SearchResult:
    return SearchResult(
        rank=1,
        document_name=DOCUMENT_NAME,
        page_number=page_number,
        text=text,
        semantic_score=score,
        keyword_bonus=0.0,
        final_score=score,
        score=score,
    )


def _sufficient_response(query: str, page_number: int, text: str) -> SearchResponse:
    return SearchResponse(
        query=query,
        insufficient_evidence=False,
        results=[_make_result(page_number, text)],
    )


def _insufficient_response(query: str) -> SearchResponse:
    return SearchResponse(query=query, insufficient_evidence=True, results=[])


def test_out_of_scope_query_never_calls_rag_agent() -> None:
    query = "오늘 서울 날씨는 어떤가요?"
    stub_agent = StubRagAgent(_sufficient_response(query, 4, "관련 없는 텍스트"))

    result = run_pipeline(query, stub_agent)

    assert stub_agent.search_called is False
    assert result["supervisor"].in_scope is False
    assert result["rag"] is None
    assert result["maintenance"] is None


def test_manual_query_calls_rag_only_and_maintenance_stays_none() -> None:
    query = "베어링 그리스는 어떻게 보충해야 하나요?"
    stub_agent = StubRagAgent(_sufficient_response(query, 8, "그리스 주입 방법을 안내합니다."))

    result = run_pipeline(query, stub_agent)

    assert stub_agent.search_called is True
    assert result["supervisor"].primary_intent == "manual_query"
    assert result["rag"] is not None
    assert result["maintenance"] is None


def test_checklist_request_with_sufficient_evidence_calls_maintenance_agent() -> None:
    query = "모터에서 진동과 소음이 심한데 점검표를 만들어 주세요."
    stub_agent = StubRagAgent(
        _sufficient_response(query, 6, "소음과 진동이 발생하면 점검하십시오.")
    )

    result = run_pipeline(query, stub_agent, equipment_id="M-202")

    assert result["equipment_id"] == "M-202"
    assert result["maintenance"] is not None
    assert result["maintenance"].checklist is not None
    checklist_items = result["maintenance"].checklist.items
    manual_items = [item for item in checklist_items if item.source_page is not None]
    assert all(item.source_page == 6 for item in manual_items)


def test_insufficient_evidence_stops_before_maintenance_agent() -> None:
    query = "모터가 과열됩니다. 점검표를 만들어 주세요."
    stub_agent = StubRagAgent(_insufficient_response(query))

    result = run_pipeline(query, stub_agent)

    assert stub_agent.search_called is True
    assert result["rag"].insufficient_evidence is True
    assert result["maintenance"] is None


def test_vibration_noise_final_message_excludes_startup_topic() -> None:
    query = "모터에서 심한 진동과 금속성 소음이 납니다. 점검 체크리스트를 만들어 주세요."
    response = SearchResponse(
        query=query,
        insufficient_evidence=False,
        results=[
            _make_result(
                6,
                "만일 반복적인 시험기동을 한다면 베어링의 이상음이 없는지 확인하고 그리스 상태를 "
                "점검하십시오.",
            ),
            _make_result(
                7,
                "기동 시 진동과 소음이 발생하는지 확인하고 커플링 정렬 상태도 점검하십시오.",
            ),
        ],
    )
    stub_agent = StubRagAgent(response)

    result = run_pipeline(query, stub_agent)

    assert "기동" not in result["final_message"]
    manual_items = [
        item for item in result["maintenance"].checklist.items if item.source_page is not None
    ]
    assert not any("기동" in item.description for item in manual_items)


def test_result_dict_has_required_top_level_keys() -> None:
    query = "모터 축 정렬 상태를 확인해 주세요."
    stub_agent = StubRagAgent(_sufficient_response(query, 5, "축 정렬 상태를 확인하십시오."))

    result = run_pipeline(query, stub_agent)

    for key in ("query", "equipment_id", "supervisor", "rag", "maintenance", "final_message", "safety_notice"):
        assert key in result
