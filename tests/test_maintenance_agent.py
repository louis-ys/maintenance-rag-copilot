"""src.maintenance_agent 단위 테스트.

실제 PDF나 임베딩 모델에 의존하지 않도록, RAG Agent가 반환할 법한
SearchResponse를 직접 구성해 Maintenance Agent의 템플릿 로직만 검증한다.
"""

from __future__ import annotations

from src.maintenance_agent import (
    MANUAL_EVIDENCE_CATEGORY,
    SAFETY_POLICY_CATEGORY,
    build_checklist,
    generate_maintenance_outputs,
)
from src.models import SearchResponse, SearchResult
from src.supervisor_agent import OUTPUT_MAINTENANCE_CHECKLIST, OUTPUT_WORK_ORDER

DOCUMENT_NAME = "hyosung_motor_manual_ko.pdf"


def _make_result(rank: int, page_number: int, text: str, score: float = 0.5) -> SearchResult:
    return SearchResult(
        rank=rank,
        document_name=DOCUMENT_NAME,
        page_number=page_number,
        text=text,
        semantic_score=score,
        keyword_bonus=0.0,
        final_score=score,
        score=score,
    )


def test_insufficient_evidence_produces_no_documents() -> None:
    response = SearchResponse(
        query="오늘 서울 날씨는 어떤가요?",
        insufficient_evidence=True,
        results=[_make_result(1, 4, "설치 장소의 주위 온도 조건")],
    )

    result = generate_maintenance_outputs(
        query=response.query,
        equipment_id="M-101",
        rag_response=response,
        requested_outputs=[OUTPUT_MAINTENANCE_CHECKLIST, OUTPUT_WORK_ORDER],
    )

    assert result is None


def test_checklist_includes_fixed_safety_policy_items() -> None:
    response = SearchResponse(
        query="베어링 그리스는 어떻게 보충해야 하나요?",
        insufficient_evidence=False,
        results=[_make_result(1, 8, "4.2.1 그리스 주입 방법 1) 전동기의 운전을 정지한다.")],
    )

    checklist = build_checklist(response)
    safety_items = [item for item in checklist.items if item.category == SAFETY_POLICY_CATEGORY]

    assert {"설비 정지", "전원 차단", "담당자 안전절차 확인"} == {item.description for item in safety_items}
    for item in safety_items:
        assert item.source_document is None
        assert item.source_page is None


def test_checklist_manual_evidence_item_has_source_page() -> None:
    response = SearchResponse(
        query="베어링 그리스는 어떻게 보충해야 하나요?",
        insufficient_evidence=False,
        results=[_make_result(1, 8, "그리스 주입 방법과 윤활 절차를 안내합니다.")],
    )

    checklist = build_checklist(response)
    manual_items = [item for item in checklist.items if item.category == MANUAL_EVIDENCE_CATEGORY]

    assert len(manual_items) >= 1
    grease_item = next(item for item in manual_items if "그리스" in item.description)
    assert grease_item.source_document == DOCUMENT_NAME
    assert grease_item.source_page == 8


def test_checklist_deduplicates_same_rule_across_multiple_results() -> None:
    response = SearchResponse(
        query="모터에서 소음과 진동이 발생합니다.",
        insufficient_evidence=False,
        results=[
            _make_result(1, 6, "소음과 진동이 발생하면 즉시 점검하십시오."),
            _make_result(2, 7, "이음이 발생하는 것은 정상음일 수 있습니다."),
        ],
    )

    checklist = build_checklist(response)
    noise_items = [item for item in checklist.items if "소음" in item.description]

    assert len(noise_items) == 1
    assert noise_items[0].source_page == 6


def test_checklist_uses_non_committal_language() -> None:
    response = SearchResponse(
        query="모터가 과열됩니다.",
        insufficient_evidence=False,
        results=[_make_result(1, 6, "과열 방지를 위해 냉각과 통풍 상태를 확인하십시오.")],
    )

    checklist = build_checklist(response)
    manual_items = [item for item in checklist.items if item.category == MANUAL_EVIDENCE_CATEGORY]

    assert manual_items
    for item in manual_items:
        assert "확인 필요" in item.description
        assert "교체" not in item.description


def test_work_order_requested_produces_draft_with_expected_fields() -> None:
    response = SearchResponse(
        query="모터가 과열됩니다. 점검표와 작업지시서를 작성해 주세요.",
        insufficient_evidence=False,
        results=[_make_result(1, 6, "과열 시 냉각과 통풍 상태를 점검하십시오.")],
    )

    result = generate_maintenance_outputs(
        query=response.query,
        equipment_id="M-202",
        rag_response=response,
        requested_outputs=[OUTPUT_MAINTENANCE_CHECKLIST, OUTPUT_WORK_ORDER],
    )

    assert result is not None
    assert result.checklist is not None
    assert result.work_order is not None
    assert result.work_order.equipment_id == "M-202"
    assert result.work_order.status == "담당자 검토 대기"
    assert result.work_order.reported_symptom == response.query
    assert any(f"{DOCUMENT_NAME} p.6" == source for source in result.work_order.reference_sources)
    assert "담당자" in result.review_required_notice


def test_vibration_noise_checklist_includes_only_relevant_items() -> None:
    """진동·소음 질문에서는 소음/진동, 베어링 또는 그리스, 정렬 항목만 매뉴얼 근거로 포함되어야 한다."""

    response = SearchResponse(
        query="모터에서 심한 진동과 금속성 소음이 납니다. 점검 체크리스트를 만들어 주세요.",
        insufficient_evidence=False,
        results=[
            _make_result(
                1,
                6,
                "진동과 소음이 발생하면 베어링 이상음 여부와 그리스 상태를 확인하고, "
                "커플링 정렬 상태도 점검하십시오.",
            ),
            _make_result(
                2,
                7,
                "부하기기와의 연결 및 축 정렬 상태, 베어링 윤활 상태, 냉각 통풍 상태, "
                "전압 상태를 함께 확인하십시오.",
            ),
            _make_result(
                3,
                9,
                "전원 및 결선 상태와 냉각 상태, 부하 조건이 정격 범위인지 확인하십시오.",
            ),
        ],
    )

    checklist = build_checklist(response)
    manual_items = [item for item in checklist.items if item.category == MANUAL_EVIDENCE_CATEGORY]
    descriptions = [item.description for item in manual_items]

    assert any("소음" in d and "진동" in d for d in descriptions)
    assert any("베어링" in d or "그리스" in d for d in descriptions)
    assert any("정렬" in d or "얼라인먼트" in d for d in descriptions)

    assert not any("냉각" in d or "통풍" in d for d in descriptions)
    assert not any("정격" in d or "과부하" in d for d in descriptions)
    assert not any("전원" in d or "결선" in d for d in descriptions)


def test_vibration_noise_checklist_excludes_startup_group_even_with_evidence() -> None:
    """진동·소음 질문에서는 검색 원문에 "기동"이 등장해도 기동 항목을 포함하지 않는다."""

    response = SearchResponse(
        query="모터에서 심한 진동과 금속성 소음이 납니다. 점검 체크리스트를 만들어 주세요.",
        insufficient_evidence=False,
        results=[
            _make_result(
                1,
                6,
                "만일 반복적인 시험기동을 한다면 과열을 방지하기 위해 시험기동 사이에 전동기가 "
                "냉각되어야 합니다. 베어링의 이상음이 없는지 확인하고 그리스 상태를 점검하십시오.",
            ),
            _make_result(
                2,
                7,
                "기동 시 진동과 소음이 발생하는지 확인하고 커플링 정렬 상태도 점검하십시오.",
            ),
        ],
    )

    checklist = build_checklist(response)
    descriptions = [item.description for item in checklist.items]

    assert not any("기동" in description for description in descriptions)
    assert "기동" not in ", ".join(checklist.matched_topics)


def test_overheat_question_includes_cooling_ventilation_item() -> None:
    response = SearchResponse(
        query="모터가 과열됩니다. 점검표를 작성해 주세요.",
        insufficient_evidence=False,
        results=[_make_result(1, 6, "과열 시 냉각과 통풍 상태를 점검하십시오.")],
    )

    checklist = build_checklist(response)
    manual_items = [item for item in checklist.items if item.category == MANUAL_EVIDENCE_CATEGORY]
    descriptions = [item.description for item in manual_items]

    assert any("냉각" in d or "통풍" in d for d in descriptions)


def test_voltage_question_includes_power_wiring_item() -> None:
    response = SearchResponse(
        query="모터 전압이 불안정합니다. 점검표를 작성해 주세요.",
        insufficient_evidence=False,
        results=[_make_result(1, 5, "전압 상태와 결선을 점검하십시오.")],
    )

    checklist = build_checklist(response)
    manual_items = [item for item in checklist.items if item.category == MANUAL_EVIDENCE_CATEGORY]
    descriptions = [item.description for item in manual_items]

    assert any("전원" in d or "결선" in d for d in descriptions)


def test_work_order_not_generated_when_not_requested() -> None:
    response = SearchResponse(
        query="베어링 그리스는 어떻게 보충해야 하나요?",
        insufficient_evidence=False,
        results=[_make_result(1, 8, "그리스 주입 방법을 안내합니다.")],
    )

    result = generate_maintenance_outputs(
        query=response.query,
        equipment_id="M-101",
        rag_response=response,
        requested_outputs=[OUTPUT_MAINTENANCE_CHECKLIST],
    )

    assert result is not None
    assert result.checklist is not None
    assert result.work_order is None
