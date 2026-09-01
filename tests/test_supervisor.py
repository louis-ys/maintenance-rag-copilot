"""src.supervisor_agent 단위 테스트."""

from __future__ import annotations

from src.supervisor_agent import (
    INTENT_MAINTENANCE_REQUEST,
    INTENT_MANUAL_QUERY,
    INTENT_OUT_OF_SCOPE,
    MAINTENANCE_AGENT_NAME,
    OUTPUT_MAINTENANCE_CHECKLIST,
    OUTPUT_WORK_ORDER,
    RAG_AGENT_NAME,
    classify_request,
)


def test_weather_question_is_out_of_scope() -> None:
    decision = classify_request("오늘 서울 날씨는 어떤가요?")

    assert decision.in_scope is False
    assert decision.primary_intent == INTENT_OUT_OF_SCOPE
    assert decision.requested_outputs == []
    assert decision.agents_to_call == []


def test_unrelated_topics_are_out_of_scope() -> None:
    for query in ["점심 메뉴 추천해줘", "여행지 추천 부탁해", "오늘 주식 시장 어때?"]:
        decision = classify_request(query)
        assert decision.in_scope is False
        assert decision.primary_intent == INTENT_OUT_OF_SCOPE


def test_general_motor_question_is_manual_query() -> None:
    decision = classify_request("베어링 그리스는 어떻게 관리하나요?")

    assert decision.in_scope is True
    assert decision.primary_intent == INTENT_MANUAL_QUERY
    assert decision.requested_outputs == ["manual_answer"]
    assert decision.agents_to_call == [RAG_AGENT_NAME]


def test_checklist_request_triggers_maintenance_checklist_output() -> None:
    decision = classify_request("모터 진동이 심한데 점검표를 만들어 주세요.")

    assert decision.in_scope is True
    assert decision.primary_intent == INTENT_MAINTENANCE_REQUEST
    assert OUTPUT_MAINTENANCE_CHECKLIST in decision.requested_outputs
    assert OUTPUT_WORK_ORDER not in decision.requested_outputs
    assert decision.agents_to_call == [RAG_AGENT_NAME, MAINTENANCE_AGENT_NAME]


def test_work_order_request_triggers_work_order_output() -> None:
    decision = classify_request("모터가 과열됩니다. 작업지시서를 작성해 주세요.")

    assert decision.in_scope is True
    assert decision.primary_intent == INTENT_MAINTENANCE_REQUEST
    assert OUTPUT_WORK_ORDER in decision.requested_outputs
    assert decision.agents_to_call == [RAG_AGENT_NAME, MAINTENANCE_AGENT_NAME]


def test_checklist_and_work_order_can_be_requested_together() -> None:
    decision = classify_request("모터가 과열됩니다. 점검표와 작업지시서를 작성해 주세요.")

    assert OUTPUT_MAINTENANCE_CHECKLIST in decision.requested_outputs
    assert OUTPUT_WORK_ORDER in decision.requested_outputs
    assert len(decision.requested_outputs) == 2


def test_maintenance_reason_has_no_typo() -> None:
    decision = classify_request("모터 진동이 심한데 점검표를 만들어 주세요.")

    assert "요청가" not in decision.reason
    assert "요청이" in decision.reason


def test_classification_is_keyword_based_not_sentence_hardcoded() -> None:
    """동일 키워드를 포함하지만 다른 문장 표현이어도 같은 분류가 나와야 한다."""

    decision_a = classify_request("전동기 축 정렬 상태를 점검하고 싶습니다.")
    decision_b = classify_request("커플링 정렬이 맞는지 전동기를 확인해 주세요.")

    assert decision_a.in_scope is True
    assert decision_b.in_scope is True
    assert decision_a.primary_intent == decision_b.primary_intent == INTENT_MANUAL_QUERY
