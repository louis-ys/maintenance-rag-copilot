"""app.py 채팅형 인터페이스(세션 상태/후속 질문 처리) 단위 테스트.

실제 임베딩 모델을 호출하지 않도록, RagAgent와 동일한 .search() 인터페이스를
가진 스텁을 주입해 messages/세션 상태 갱신 로직만 검증한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import (
    CHECKLIST_FOLLOW_UP_PROMPT,
    GREETING_MESSAGE,
    NO_PRIOR_EVIDENCE_MESSAGE,
    PENDING_ACTION_SELECT_MOTOR,
    QUICK_REPLY_WORK_ORDER_QUESTION,
    RAG_EXAMPLE_QUESTION,
    SCENARIO_EXAMPLE_QUESTION,
    append_assistant_message,
    append_user_message,
    handle_user_query,
    init_session_state,
    is_follow_up_work_order_request,
    reset_conversation,
)
from src.maintenance_agent import MANUAL_EVIDENCE_CATEGORY, ChecklistItem, MaintenanceChecklist
from src.models import SearchResponse, SearchResult
from src.orchestrator import DEFAULT_EQUIPMENT_ID

DOCUMENT_NAME = "hyosung_motor_manual_ko.pdf"


class StubRagAgent:
    """테스트용 RagAgent 대역. search() 호출 여부와 반환값을 통제한다."""

    def __init__(self, response: SearchResponse) -> None:
        self._response = response
        self.search_called = False

    def search(self, query: str, top_k: int = 3) -> SearchResponse:
        self.search_called = True
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
        page_text=text,
    )


def _make_state() -> dict:
    state: dict = {}
    init_session_state(state)
    return state


def test_init_session_state_creates_greeting_message() -> None:
    state = _make_state()

    assert len(state["messages"]) == 1
    assert state["messages"][0]["role"] == "assistant"
    assert state["messages"][0]["content"] == GREETING_MESSAGE
    assert state["equipment_id"] == DEFAULT_EQUIPMENT_ID
    assert state["last_rag_result"] is None
    assert state["last_checklist"] is None


def test_append_user_message_is_stored_in_order() -> None:
    state = _make_state()

    append_user_message(state, "베어링 그리스는 어떻게 보충해야 하나요?")

    assert state["messages"][-1] == {
        "role": "user",
        "content": "베어링 그리스는 어떻게 보충해야 하나요?",
        "detail": None,
    }


def test_append_assistant_message_is_stored_with_detail() -> None:
    state = _make_state()

    append_assistant_message(state, "답변입니다.", detail={"query": "x"})

    assert state["messages"][-1]["role"] == "assistant"
    assert state["messages"][-1]["content"] == "답변입니다."
    assert state["messages"][-1]["detail"] == {"query": "x"}


def test_handle_user_query_first_question_starts_motor_selection_then_calls_orchestrator() -> None:
    """점검표 요청(시나리오 질문)은 먼저 설비 선택을 거친 뒤에야 RAG/Maintenance Agent를 호출한다."""

    state = _make_state()
    query = "모터에서 진동과 소음이 심한데 점검표를 만들어 주세요."
    stub_agent = StubRagAgent(
        SearchResponse(
            query=query,
            insufficient_evidence=False,
            results=[_make_result(6, "소음과 진동이 발생하면 점검하십시오.")],
        )
    )

    handle_user_query(state, query, stub_agent)

    assert stub_agent.search_called is False
    assert state["pending_action"] == PENDING_ACTION_SELECT_MOTOR
    assert state["last_checklist"] is None

    handle_user_query(state, "1번", stub_agent)

    assert stub_agent.search_called is True
    assert state["last_user_query"] == query
    assert state["last_rag_result"] is not None
    assert state["last_checklist"] is not None
    assert state["pending_action"] is None
    assert len(state["messages"]) == 5  # 인사말 + 질문 + 설비목록 안내 + "1번" + 최종 답변
    assert state["messages"][-1]["role"] == "assistant"
    assert state["messages"][-1]["detail"]["maintenance"].checklist is not None


def test_follow_up_work_order_reuses_last_rag_evidence_and_checklist() -> None:
    state = _make_state()
    rag_result = SearchResponse(
        query="모터에서 진동과 소음이 심합니다.",
        insufficient_evidence=False,
        results=[_make_result(6, "소음과 진동이 발생하면 점검하십시오.")],
    )
    checklist = MaintenanceChecklist(
        items=[
            ChecklistItem(
                category=MANUAL_EVIDENCE_CATEGORY,
                description="베어링 이상음 여부 확인 필요",
                source_document=DOCUMENT_NAME,
                source_page=6,
            )
        ],
        review_notice="검토 필요",
        matched_topics=("진동·소음",),
    )
    state["last_rag_result"] = rag_result
    state["last_checklist"] = checklist

    stub_agent = StubRagAgent(rag_result)
    handle_user_query(state, "이 내용을 작업지시서로 정리해 주세요.", stub_agent)

    assert stub_agent.search_called is False  # RAG 재검색 없이 기존 근거 재사용
    last_message = state["messages"][-1]
    assert last_message["role"] == "assistant"
    detail = last_message["detail"]
    assert detail["is_follow_up"] is True
    assert detail["maintenance"].work_order is not None
    assert detail["maintenance"].work_order.inspection_items == ["베어링 이상음 여부 확인 필요"]
    assert detail["rag"] is rag_result


def test_follow_up_without_prior_evidence_is_blocked() -> None:
    state = _make_state()
    stub_agent = StubRagAgent(SearchResponse(query="x", insufficient_evidence=True, results=[]))

    handle_user_query(state, "작업지시서 형태로 만들어 주세요.", stub_agent)

    assert stub_agent.search_called is False
    last_message = state["messages"][-1]
    assert last_message["role"] == "assistant"
    assert last_message["content"] == NO_PRIOR_EVIDENCE_MESSAGE
    assert last_message["detail"] is None


def test_is_follow_up_work_order_request_detects_paraphrases() -> None:
    assert is_follow_up_work_order_request("이 내용을 작업지시서로 정리해 주세요.") is True
    assert is_follow_up_work_order_request("작업지시서 형태로 만들어 주세요.") is True
    assert is_follow_up_work_order_request("현장 작업자가 사용할 수 있게 정리해 주세요.") is True


def test_is_follow_up_work_order_request_false_for_new_symptom_question() -> None:
    # 증상 설명이 포함된 완결된 질문은 새 질문으로 처리해야 한다.
    assert is_follow_up_work_order_request("모터가 과열됩니다. 점검표와 작업지시서를 작성해 주세요.") is False
    assert is_follow_up_work_order_request("베어링 그리스는 어떻게 보충해야 하나요?") is False


def test_reset_conversation_clears_state_and_resets_equipment_id() -> None:
    state = _make_state()
    state["equipment_id"] = "M-999"
    stub_agent = StubRagAgent(SearchResponse(query="x", insufficient_evidence=True, results=[]))
    handle_user_query(state, "모터가 과열됩니다.", stub_agent)

    reset_conversation(state)

    assert len(state["messages"]) == 1
    assert state["messages"][0]["content"] == GREETING_MESSAGE
    assert state["last_user_query"] is None
    assert state["last_rag_result"] is None
    assert state["last_checklist"] is None
    assert state["last_orchestration_result"] is None
    assert state["equipment_id"] == DEFAULT_EQUIPMENT_ID


def test_example_questions_are_processed_like_typed_user_messages() -> None:
    state = _make_state()
    rag_response = SearchResponse(
        query=RAG_EXAMPLE_QUESTION,
        insufficient_evidence=False,
        results=[_make_result(8, "그리스 주입 방법을 안내합니다.")],
    )
    stub_agent = StubRagAgent(rag_response)

    handle_user_query(state, RAG_EXAMPLE_QUESTION, stub_agent)

    assert state["messages"][-2] == {"role": "user", "content": RAG_EXAMPLE_QUESTION, "detail": None}
    assert state["messages"][-1]["role"] == "assistant"
    assert stub_agent.search_called is True


def test_scenario_example_question_generates_checklist_for_follow_up() -> None:
    state = _make_state()
    rag_response = SearchResponse(
        query=SCENARIO_EXAMPLE_QUESTION,
        insufficient_evidence=False,
        results=[_make_result(6, "소음과 진동이 발생하면 베어링 이상음을 점검하십시오.")],
    )
    stub_agent = StubRagAgent(rag_response)

    handle_user_query(state, SCENARIO_EXAMPLE_QUESTION, stub_agent)
    handle_user_query(state, "1번", stub_agent)

    assert state["last_checklist"] is not None
    assert state["messages"][-1]["detail"]["maintenance"].checklist is not None


def test_scenario_checklist_answer_contains_natural_follow_up_prompt_and_no_dev_tags() -> None:
    state = _make_state()
    rag_response = SearchResponse(
        query=SCENARIO_EXAMPLE_QUESTION,
        insufficient_evidence=False,
        results=[_make_result(6, "소음과 진동이 발생하면 베어링 이상음을 점검하십시오.")],
    )
    stub_agent = StubRagAgent(rag_response)

    handle_user_query(state, SCENARIO_EXAMPLE_QUESTION, stub_agent)
    handle_user_query(state, "1번", stub_agent)

    answer = state["messages"][-1]["content"]
    assert "[매뉴얼 근거]" not in answer
    assert "[시스템 안전정책]" not in answer
    assert "확인 필요" not in answer
    assert state["show_work_order_quick_reply"] is True


def test_quick_reply_work_order_question_is_processed_as_follow_up() -> None:
    state = _make_state()
    rag_result = SearchResponse(
        query="모터에서 진동과 소음이 심합니다.",
        insufficient_evidence=False,
        results=[_make_result(6, "소음과 진동이 발생하면 점검하십시오.")],
    )
    checklist = MaintenanceChecklist(
        items=[
            ChecklistItem(
                category=MANUAL_EVIDENCE_CATEGORY,
                description="베어링 이상음 여부 확인 필요",
                source_document=DOCUMENT_NAME,
                source_page=6,
            )
        ],
        review_notice="검토 필요",
        matched_topics=("진동·소음",),
    )
    state["last_rag_result"] = rag_result
    state["last_checklist"] = checklist
    state["show_work_order_quick_reply"] = True

    stub_agent = StubRagAgent(rag_result)
    handle_user_query(state, QUICK_REPLY_WORK_ORDER_QUESTION, stub_agent)

    assert state["messages"][-2] == {
        "role": "user",
        "content": QUICK_REPLY_WORK_ORDER_QUESTION,
        "detail": None,
    }
    last_message = state["messages"][-1]
    assert last_message["role"] == "assistant"
    detail = last_message["detail"]
    assert detail["is_follow_up"] is True
    assert detail["maintenance"].work_order is not None
    assert state["show_work_order_quick_reply"] is False


def test_rag_only_answer_does_not_force_work_order_prompt() -> None:
    state = _make_state()
    rag_response = SearchResponse(
        query=RAG_EXAMPLE_QUESTION,
        insufficient_evidence=False,
        results=[_make_result(8, "그리스 주입 방법을 안내합니다.")],
    )
    stub_agent = StubRagAgent(rag_response)

    handle_user_query(state, RAG_EXAMPLE_QUESTION, stub_agent)

    answer = state["messages"][-1]["content"]
    assert CHECKLIST_FOLLOW_UP_PROMPT not in answer
    assert state["show_work_order_quick_reply"] is False


def test_detail_payload_keeps_raw_technical_data_for_expander() -> None:
    state = _make_state()
    rag_response = SearchResponse(
        query=SCENARIO_EXAMPLE_QUESTION,
        insufficient_evidence=False,
        results=[_make_result(6, "소음과 진동이 발생하면 베어링 이상음을 점검하십시오.")],
    )
    stub_agent = StubRagAgent(rag_response)

    handle_user_query(state, SCENARIO_EXAMPLE_QUESTION, stub_agent)
    handle_user_query(state, "1번", stub_agent)

    detail = state["messages"][-1]["detail"]
    assert detail["rag"] is rag_response
    assert detail["rag"].results[0].semantic_score == rag_response.results[0].semantic_score
    checklist_items = detail["maintenance"].checklist.items
    assert any(item.category == MANUAL_EVIDENCE_CATEGORY for item in checklist_items)


def test_new_conversation_resets_quick_reply_state() -> None:
    state = _make_state()
    rag_response = SearchResponse(
        query=SCENARIO_EXAMPLE_QUESTION,
        insufficient_evidence=False,
        results=[_make_result(6, "소음과 진동이 발생하면 베어링 이상음을 점검하십시오.")],
    )
    stub_agent = StubRagAgent(rag_response)
    handle_user_query(state, SCENARIO_EXAMPLE_QUESTION, stub_agent)
    handle_user_query(state, "1번", stub_agent)
    assert state["show_work_order_quick_reply"] is True

    reset_conversation(state)

    assert state["show_work_order_quick_reply"] is False


# ────────────────────────────
# 시나리오 멀티턴 설비 선택(JSON 기반) 테스트
# ────────────────────────────

VIBRATION_NOISE_QUERY = "모터에서 심한 진동과 금속성 소음이 납니다. 점검 체크리스트를 만들어 주세요."


def _vibration_noise_rag_response() -> SearchResponse:
    return SearchResponse(
        query=VIBRATION_NOISE_QUERY,
        insufficient_evidence=False,
        results=[
            _make_result(
                6,
                "진동과 소음이 발생하면 베어링 이상음 여부와 그리스 상태를 확인하고, "
                "커플링 정렬 상태도 점검하십시오.",
            ),
            _make_result(
                7,
                "부하기기와의 연결 및 축 정렬 상태, 베어링 윤활 상태, 냉각 통풍 상태, "
                "전압 상태를 함께 확인하십시오.",
            ),
            _make_result(
                9,
                "전원 및 결선 상태와 냉각 상태, 부하 조건이 정격 범위인지 확인하십시오.",
            ),
        ],
    )


def test_first_scenario_question_returns_equipment_list() -> None:
    state = _make_state()
    stub_agent = StubRagAgent(_vibration_noise_rag_response())

    handle_user_query(state, VIBRATION_NOISE_QUERY, stub_agent)

    answer = state["messages"][-1]["content"]
    assert "A모터" in answer
    assert "B모터" in answer
    assert "C모터" in answer
    assert "M-101" in answer
    assert "M-102" in answer
    assert "M-103" in answer


def test_first_scenario_question_does_not_call_rag_agent() -> None:
    state = _make_state()
    stub_agent = StubRagAgent(_vibration_noise_rag_response())

    handle_user_query(state, VIBRATION_NOISE_QUERY, stub_agent)

    assert stub_agent.search_called is False


def test_first_scenario_question_does_not_call_maintenance_agent() -> None:
    state = _make_state()
    stub_agent = StubRagAgent(_vibration_noise_rag_response())

    handle_user_query(state, VIBRATION_NOISE_QUERY, stub_agent)

    assert state["last_checklist"] is None
    assert state["messages"][-1]["detail"] is None


def test_selecting_motor_three_stores_c_motor() -> None:
    state = _make_state()
    stub_agent = StubRagAgent(_vibration_noise_rag_response())
    handle_user_query(state, VIBRATION_NOISE_QUERY, stub_agent)

    handle_user_query(state, "3번", stub_agent)

    selected_motor = state["selected_motor"]
    assert selected_motor is not None
    assert selected_motor.alias == "C모터"
    assert selected_motor.id == "M-103"


def test_selecting_motor_changes_equipment_id_to_m103() -> None:
    state = _make_state()
    stub_agent = StubRagAgent(_vibration_noise_rag_response())
    handle_user_query(state, VIBRATION_NOISE_QUERY, stub_agent)

    handle_user_query(state, "3번", stub_agent)

    assert state["equipment_id"] == "M-103"


def test_selecting_motor_calls_rag_agent() -> None:
    state = _make_state()
    stub_agent = StubRagAgent(_vibration_noise_rag_response())
    handle_user_query(state, VIBRATION_NOISE_QUERY, stub_agent)

    handle_user_query(state, "3번", stub_agent)

    assert stub_agent.search_called is True


def test_selecting_motor_calls_maintenance_agent() -> None:
    state = _make_state()
    stub_agent = StubRagAgent(_vibration_noise_rag_response())
    handle_user_query(state, VIBRATION_NOISE_QUERY, stub_agent)

    handle_user_query(state, "3번", stub_agent)

    assert state["last_checklist"] is not None
    assert state["messages"][-1]["detail"]["maintenance"].checklist is not None


def test_final_answer_includes_selected_motor_identity() -> None:
    state = _make_state()
    stub_agent = StubRagAgent(_vibration_noise_rag_response())
    handle_user_query(state, VIBRATION_NOISE_QUERY, stub_agent)

    handle_user_query(state, "3번", stub_agent)

    answer = state["messages"][-1]["content"]
    assert "C모터" in answer
    assert "컨베이어 구동 모터" in answer
    assert "M-103" in answer


def test_final_answer_includes_matched_checklist_topics() -> None:
    state = _make_state()
    stub_agent = StubRagAgent(_vibration_noise_rag_response())
    handle_user_query(state, VIBRATION_NOISE_QUERY, stub_agent)

    handle_user_query(state, "3번", stub_agent)

    answer = state["messages"][-1]["content"]
    assert "베어링·윤활" in answer
    assert "진동·소음" in answer
    assert "정렬" in answer


def test_invalid_motor_number_keeps_pending_action() -> None:
    state = _make_state()
    stub_agent = StubRagAgent(_vibration_noise_rag_response())
    handle_user_query(state, VIBRATION_NOISE_QUERY, stub_agent)

    handle_user_query(state, "7번", stub_agent)

    assert state["pending_action"] == PENDING_ACTION_SELECT_MOTOR
    assert state["selected_motor"] is None
    assert stub_agent.search_called is False
    answer = state["messages"][-1]["content"]
    assert "다시 선택" in answer


def test_rag_only_question_never_triggers_motor_selection() -> None:
    state = _make_state()
    rag_response = SearchResponse(
        query=RAG_EXAMPLE_QUESTION,
        insufficient_evidence=False,
        results=[_make_result(8, "그리스 주입 방법을 안내합니다.")],
    )
    stub_agent = StubRagAgent(rag_response)

    handle_user_query(state, RAG_EXAMPLE_QUESTION, stub_agent)

    assert state["pending_action"] is None
    assert state["selected_motor"] is None
    assert stub_agent.search_called is True


def test_new_conversation_resets_motor_selection_state() -> None:
    state = _make_state()
    stub_agent = StubRagAgent(_vibration_noise_rag_response())
    handle_user_query(state, VIBRATION_NOISE_QUERY, stub_agent)
    assert state["pending_action"] == PENDING_ACTION_SELECT_MOTOR

    reset_conversation(state)

    assert state["pending_action"] is None
    assert state["pending_user_query"] is None
    assert state["pending_intent"] is None
    assert state["pending_supervisor_result"] is None
    assert state["selected_motor"] is None


def test_quick_select_button_message_is_saved_as_user_message() -> None:
    """설비 빠른 선택 버튼은 "N번" 텍스트를 실제 사용자 메시지로 handle_user_query에 전달한다."""

    state = _make_state()
    stub_agent = StubRagAgent(_vibration_noise_rag_response())
    handle_user_query(state, VIBRATION_NOISE_QUERY, stub_agent)

    handle_user_query(state, "3번", stub_agent)

    user_messages = [message for message in state["messages"] if message["role"] == "user"]
    assert user_messages[-1] == {"role": "user", "content": "3번", "detail": None}


# --- Streamlit AppTest 회귀 테스트 -------------------------------------------------
#
# equipment_id를 st.text_input(key="equipment_id")로 사용하면, 설비 선택 처리 중
# state["equipment_id"]를 갱신할 때 "cannot be modified after the widget with key
# equipment_id is instantiated" StreamlitAPIException이 발생한다. 아래 테스트는 실제
# 앱을 렌더링해 이 예외가 재발하지 않는지 검증한다.

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def _run_app() -> AppTest:
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    assert not at.exception
    return at


def test_equipment_id_is_not_bound_to_any_widget_key() -> None:
    """equipment_id는 애플리케이션 상태로만 쓰이고, 어떤 위젯의 key로도 쓰이지 않는다."""

    at = _run_app()

    widget_keys = {widget.key for widget in (*at.text_input, *at.button)}
    assert "equipment_id" not in widget_keys


def test_motor_quick_select_does_not_raise_widget_exception() -> None:
    """사이드바 위젯 인스턴스화 이후 equipment_id를 갱신해도 StreamlitAPIException이 발생하지 않는다."""

    at = _run_app()

    at.chat_input[0].set_value(SCENARIO_EXAMPLE_QUESTION).run()
    assert not at.exception

    quick_select_button = next(button for button in at.button if button.key == "quick_select_motor_3")
    quick_select_button.click().run()

    assert not at.exception
    assert at.session_state["equipment_id"] == "M-103"
    assert at.session_state["selected_motor"].id == "M-103"


def test_new_conversation_button_resets_selected_motor_and_equipment_id_in_app() -> None:
    """"새 대화" 버튼은 selected_motor와 equipment_id, 설비 선택 대기 상태를 모두 초기화하고
    사이드바에는 "선택된 설비: 없음"이 다시 표시된다."""

    at = _run_app()

    at.chat_input[0].set_value(SCENARIO_EXAMPLE_QUESTION).run()
    quick_select_button = next(button for button in at.button if button.key == "quick_select_motor_3")
    quick_select_button.click().run()
    assert at.session_state["equipment_id"] == "M-103"

    new_chat_button = next(button for button in at.button if button.label == "새 대화")
    new_chat_button.click().run()

    assert not at.exception
    assert at.session_state["selected_motor"] is None
    assert at.session_state["equipment_id"] == DEFAULT_EQUIPMENT_ID
    assert at.session_state["pending_action"] is None
    assert at.session_state["pending_user_query"] is None
    assert at.session_state["pending_intent"] is None
    assert at.session_state["pending_supervisor_result"] is None

    sidebar_captions = [caption.value for caption in at.sidebar.caption]
    assert "선택된 설비: 없음" in sidebar_captions


def test_rag_only_question_renders_without_exception_in_app() -> None:
    at = _run_app()

    at.chat_input[0].set_value(RAG_EXAMPLE_QUESTION).run()

    assert not at.exception
    assert at.session_state["pending_action"] is None
