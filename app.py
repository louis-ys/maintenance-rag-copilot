"""효성 저압전동기 유지보수 AI 코파일럿 Streamlit 채팅 화면.

Supervisor Agent -> RAG Agent -> Maintenance Agent 순서로 실행되는
오케스트레이터 결과를 채팅 답변으로 자연스럽게 보여주고, 상세 기술 정보는
Assistant 답변 아래 expander에 접어서 배치한다.
"""

from __future__ import annotations

import html
import sys
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src.equipment_repository import EquipmentRepository, Motor
from src.evidence_snippet import extract_core_evidence
from src.maintenance_agent import (
    MANUAL_EVIDENCE_CATEGORY,
    SAFETY_POLICY_CATEGORY,
    MaintenanceChecklist,
    MaintenanceResult,
    REVIEW_NOTICE,
    build_work_order,
)
from src.orchestrator import DEFAULT_EQUIPMENT_ID, SAFETY_NOTICE, run_pipeline
from src.rag_agent import RagAgent
from src.source_text_formatter import DISPLAY_NOTICE, format_page_text_for_display
from src.supervisor_agent import (
    MOTOR_SCOPE_KEYWORDS,
    OUTPUT_MAINTENANCE_CHECKLIST,
    OUTPUT_WORK_ORDER,
    SupervisorDecision,
    classify_request,
)

PAGE_TITLE = "효성 저압전동기 유지보수 AI 코파일럿"

# 내부 파일명(hyosung_motor_manual_ko.pdf)은 그대로 두고, 화면에는 정식 문서명만 보여준다.
DOCUMENT_DISPLAY_NAME = "효성 저압전동기 취급설명서"

_INTERNAL_DOCUMENT_NAME = "hyosung_motor_manual_ko.pdf"

GREETING_MESSAGE = (
    "안녕하세요. 효성 저압전동기 유지보수 AI 코파일럿입니다.\n"
    "전동기 증상이나 매뉴얼 관련 질문을 입력해 주세요."
)

RAG_EXAMPLE_QUESTION = "베어링 그리스는 어떻게 보충해야 하나요?"
SCENARIO_EXAMPLE_QUESTION = "모터에서 심한 진동과 금속성 소음이 납니다. 점검 체크리스트를 만들어 주세요."

NO_PRIOR_EVIDENCE_MESSAGE = "먼저 설비 증상이나 점검 요청을 입력해 주세요."
FOLLOW_UP_FINAL_MESSAGE = (
    "이전 검색 근거와 점검 체크리스트를 바탕으로 작업지시서 초안을 정리했습니다. "
    "최종 정비 판단과 승인은 설비보전 담당자가 수행해야 합니다."
)

# 시나리오 체크리스트 답변 마지막에 붙이는 자연어 후속 안내(개발용 문구 대신 사용).
CHECKLIST_FOLLOW_UP_PROMPT = "이 체크리스트를 현장 작업자가 사용할 수 있는 작업지시서로 정리해드릴까요?"
CHECKLIST_INTRO_DEFAULT = "요청하신 점검 체크리스트를 정리했습니다."

WORK_ORDER_INTRO = "앞서 확인한 매뉴얼 검색 근거와 점검 체크리스트를 바탕으로 작업지시서를 정리했습니다."
WORK_ORDER_POST_CHECK_NOTICE = "점검 결과를 기록하고 이상이 발견되면 담당자에게 즉시 보고해 주세요."

# "작업지시서로 정리" 빠른 응답 버튼을 누르면 실제 사용자 메시지로 추가되는 질문.
QUICK_REPLY_WORK_ORDER_LABEL = "작업지시서로 정리"
QUICK_REPLY_WORK_ORDER_QUESTION = "이 내용을 현장 작업자가 사용할 수 있는 작업지시서로 정리해 주세요."

# 후속 질문("이 내용을 작업지시서로 정리해 주세요" 등)을 판단하는 키워드.
# 증상 설명(모터/전동기 등 업무 키워드)이 포함되면 새로운 질문으로 보고
# 후속 요청으로 판단하지 않는다.
FOLLOW_UP_ACTION_KEYWORDS: tuple[str, ...] = ("정리해", "만들어", "작성해")
FOLLOW_UP_WORK_ORDER_TARGET_KEYWORDS: tuple[str, ...] = ("작업지시서", "작업 지시서", "지시서")
FOLLOW_UP_FIELD_WORKER_KEYWORDS: tuple[str, ...] = ("현장", "작업자")

# 시나리오(점검표/작업지시서 요청) 첫 질문에서 RAG/Maintenance Agent를 바로 호출하지 않고
# JSON 설비 목록에서 담당 설비를 먼저 선택받는 중간 단계에서 사용하는 상태값.
PENDING_ACTION_SELECT_MOTOR = "select_motor"

EQUIPMENT_JSON_PATH_LABEL = "data/equipment/motors.json"
EQUIPMENT_SOURCE_NOTE = "(시연용 가상 설비 데이터)"
MANUAL_SOURCE_LABEL = "효성 저압전동기 취급설명서 PDF (RAG 검색 결과)"

EQUIPMENT_SELECT_PROMPT_HEADER = "현재 등록된 담당 모터는 다음과 같습니다.\n어느 모터에서 문제가 발생하고 있나요?"
EQUIPMENT_SELECT_PROMPT_FOOTER = "번호 또는 모터명을 입력해 주세요."
EQUIPMENT_RESELECT_PROMPT_HEADER = "등록된 모터 번호를 다시 선택해 주세요."

SCENARIO_SOURCE_NOTICE = (
    "설비 정보는 시연용 JSON 데이터에서 조회했으며,\n"
    "점검항목은 효성 저압전동기 취급설명서 RAG 검색 결과를 사용했습니다."
)
SCENARIO_REVIEW_NOTICE = "본 체크리스트는 초안이며\n설비보전 담당자의 최종 검토와 승인이 필요합니다."


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _display_document_name(text: str) -> str:
    """내부 파일명이 포함된 문자열을 화면 표시용 문서명으로 바꾼다."""

    return text.replace(_INTERNAL_DOCUMENT_NAME, DOCUMENT_DISPLAY_NAME)


@st.cache_resource(show_spinner="매뉴얼 인덱스를 불러오는 중입니다...")
def get_rag_agent() -> RagAgent:
    agent = RagAgent()
    agent.build_index()
    return agent


def get_equipment_repository() -> EquipmentRepository:
    """설비 JSON 저장소를 로딩한다. 임베딩/벡터 검색 없이 일반 JSON 조회만 사용한다."""

    return EquipmentRepository()


def _is_scenario_request(supervisor_decision: SupervisorDecision) -> bool:
    """점검표/작업지시서가 필요한 시나리오 질문인지 판단한다."""

    if not supervisor_decision.in_scope:
        return False
    return bool(
        set(supervisor_decision.requested_outputs) & {OUTPUT_MAINTENANCE_CHECKLIST, OUTPUT_WORK_ORDER}
    )


def is_follow_up_work_order_request(text: str) -> bool:
    """직전 근거를 재사용해 작업지시서로 정리해 달라는 후속 요청인지 판단한다."""

    if _contains_any(text, MOTOR_SCOPE_KEYWORDS):
        return False
    if not _contains_any(text, FOLLOW_UP_ACTION_KEYWORDS):
        return False
    if _contains_any(text, FOLLOW_UP_WORK_ORDER_TARGET_KEYWORDS):
        return True
    return all(keyword in text for keyword in FOLLOW_UP_FIELD_WORKER_KEYWORDS)


def create_greeting_message() -> dict[str, Any]:
    return {"role": "assistant", "content": GREETING_MESSAGE, "detail": None}


def init_session_state(state: MutableMapping[str, Any]) -> None:
    if "messages" not in state:
        state["messages"] = [create_greeting_message()]
    state.setdefault("equipment_id", DEFAULT_EQUIPMENT_ID)
    state.setdefault("last_user_query", None)
    state.setdefault("last_rag_result", None)
    state.setdefault("last_checklist", None)
    state.setdefault("last_orchestration_result", None)
    state.setdefault("show_work_order_quick_reply", False)
    state.setdefault("pending_action", None)
    state.setdefault("pending_user_query", None)
    state.setdefault("pending_intent", None)
    state.setdefault("pending_supervisor_result", None)
    state.setdefault("selected_motor", None)


def reset_conversation(state: MutableMapping[str, Any]) -> None:
    state["messages"] = [create_greeting_message()]
    state["equipment_id"] = DEFAULT_EQUIPMENT_ID
    state["last_user_query"] = None
    state["last_rag_result"] = None
    state["last_checklist"] = None
    state["last_orchestration_result"] = None
    state["show_work_order_quick_reply"] = False
    state["pending_action"] = None
    state["pending_user_query"] = None
    state["pending_intent"] = None
    state["pending_supervisor_result"] = None
    state["selected_motor"] = None


def append_user_message(state: MutableMapping[str, Any], text: str) -> None:
    state["messages"].append({"role": "user", "content": text, "detail": None})


def append_assistant_message(
    state: MutableMapping[str, Any], content: str, detail: dict[str, Any] | None = None
) -> None:
    state["messages"].append({"role": "assistant", "content": content, "detail": detail})


def _build_follow_up_detail(
    query: str, equipment_id: str, last_rag_result: Any, last_checklist: Any
) -> dict[str, Any]:
    work_order = build_work_order(query, equipment_id, last_checklist)
    maintenance = MaintenanceResult(
        checklist=last_checklist, work_order=work_order, review_required_notice=REVIEW_NOTICE
    )
    return {
        "query": query,
        "equipment_id": equipment_id,
        "supervisor": None,
        "rag": last_rag_result,
        "maintenance": maintenance,
        "final_message": FOLLOW_UP_FINAL_MESSAGE,
        "safety_notice": SAFETY_NOTICE,
        "is_follow_up": True,
    }


def _naturalize_checklist_description(description: str) -> str:
    """체크리스트 항목 설명에서 '확인 필요'류 개발용 표현을 자연스러운 문장으로 바꾼다."""

    for suffix in (" 여부 확인 필요", " 확인 필요"):
        if description.endswith(suffix):
            return description[: -len(suffix)] + " 확인"
    return description


def _checklist_display_lines(checklist: MaintenanceChecklist) -> list[str]:
    """카테고리 라벨 없이, 사람이 읽기 자연스러운 순서의 체크리스트 문장 목록을 만든다."""

    safety_descs = [item.description for item in checklist.items if item.category == SAFETY_POLICY_CATEGORY]
    manual_descs = [item.description for item in checklist.items if item.category == MANUAL_EVIDENCE_CATEGORY]

    lines: list[str] = []
    if "설비 정지" in safety_descs and "전원 차단" in safety_descs:
        lines.append("설비 정지 및 전원 차단")
        lines.extend(desc for desc in safety_descs if desc not in ("설비 정지", "전원 차단"))
    else:
        lines.extend(safety_descs)
    lines.extend(_naturalize_checklist_description(desc) for desc in manual_descs)
    return lines


def _build_checklist_answer(checklist: MaintenanceChecklist) -> str:
    topic_label = ", ".join(checklist.matched_topics)
    intro = f"요청하신 {topic_label} 점검 체크리스트를 정리했습니다." if topic_label else CHECKLIST_INTRO_DEFAULT

    lines = [intro, ""]
    lines.extend(
        f"{index}. {text}" for index, text in enumerate(_checklist_display_lines(checklist), start=1)
    )
    lines.append("")
    lines.append(CHECKLIST_FOLLOW_UP_PROMPT)
    return "\n".join(lines)


def _format_equipment_select_list(motors: list[Motor]) -> str:
    lines = [f"{motor.number}. {motor.alias} — {motor.name}({motor.id}) / {motor.location}" for motor in motors]
    return "\n".join(lines)


def _format_equipment_reselect_list(motors: list[Motor]) -> str:
    lines = [f"{motor.number}번 {motor.alias} — {motor.name}" for motor in motors]
    return "\n".join(lines)


def _build_equipment_select_message(motors: list[Motor]) -> str:
    return "\n\n".join(
        [EQUIPMENT_SELECT_PROMPT_HEADER, _format_equipment_select_list(motors), EQUIPMENT_SELECT_PROMPT_FOOTER]
    )


def _build_equipment_reselect_message(motors: list[Motor]) -> str:
    return "\n\n".join([EQUIPMENT_RESELECT_PROMPT_HEADER, _format_equipment_reselect_list(motors)])


def _format_equipment_info(motor: Motor) -> str:
    lines = [
        "설비 정보",
        f"- 설비명: {motor.name}",
        f"- 설비 ID: {motor.id}",
        f"- 구분명: {motor.alias}",
        f"- 설치 위치: {motor.location}",
        f"- 모델: {motor.model}",
        f"- 현재 상태: {motor.status}",
    ]
    return "\n".join(lines)


def _build_scenario_selection_answer(motor: Motor, checklist: MaintenanceChecklist) -> str:
    """설비 선택 후 설비 정보(JSON)와 점검 체크리스트(RAG 근거)를 구분해 보여주는 채팅 답변을 만든다."""

    topic_label = ", ".join(checklist.matched_topics)
    checklist_intro = (
        f"요청하신 {motor.name}의 {topic_label} 점검 체크리스트를 정리했습니다."
        if topic_label
        else f"요청하신 {motor.name}의 점검 체크리스트를 정리했습니다."
    )

    lines = [f"{motor.alias}인 {motor.name}({motor.id})를 선택했습니다.", ""]
    lines.append(_format_equipment_info(motor))
    lines.append("")
    lines.append(checklist_intro)
    lines.append("")
    lines.extend(
        f"{index}. {text}" for index, text in enumerate(_checklist_display_lines(checklist), start=1)
    )
    if topic_label:
        lines.append("")
        lines.append(f"점검 주제:\n{topic_label}")
    lines.append("")
    lines.append(SCENARIO_SOURCE_NOTICE)
    lines.append("")
    lines.append(SCENARIO_REVIEW_NOTICE)
    return "\n".join(lines)


def _build_work_order_answer(maintenance: MaintenanceResult) -> str:
    work_order = maintenance.work_order
    assert work_order is not None

    lines = [WORK_ORDER_INTRO, ""]
    lines.append(f"- 작업명: {_display_document_name(work_order.title)}")
    lines.append(f"- 설비 ID: {work_order.equipment_id}")
    lines.append(f"- 발생 증상: {work_order.reported_symptom}")
    lines.append(f"- 작업 목적: {work_order.purpose}")
    if work_order.inspection_items:
        lines.append("- 작업 순서:")
        lines.extend(
            f"  {index}. {_naturalize_checklist_description(item)}"
            for index, item in enumerate(work_order.inspection_items, start=1)
        )
    lines.append(f"- 작업 후 확인사항: {WORK_ORDER_POST_CHECK_NOTICE}")
    lines.append(f"- 담당자 승인 필요 안내: {maintenance.review_required_notice}")
    return "\n".join(lines)


def _build_chat_answer(detail: dict[str, Any]) -> str:
    supervisor = detail.get("supervisor")
    rag = detail.get("rag")
    maintenance = detail.get("maintenance")
    final_message = _display_document_name(detail["final_message"])

    if supervisor is not None and not supervisor.in_scope:
        return final_message
    if rag is None or rag.insufficient_evidence:
        return final_message

    if maintenance is not None and maintenance.work_order is not None:
        return _build_work_order_answer(maintenance)

    if maintenance is not None and maintenance.checklist is not None:
        return _build_checklist_answer(maintenance.checklist)

    top_result = rag.results[0]
    core_evidence = extract_core_evidence(rag.query, top_result.page_text or top_result.text)
    return "\n".join([core_evidence or top_result.text, "", final_message])


def _start_motor_selection(state: MutableMapping[str, Any], query: str, supervisor_decision: SupervisorDecision) -> None:
    """시나리오 질문(점검표/작업지시서 요청)에서 RAG/Maintenance Agent 호출 전에
    JSON 설비 목록을 먼저 안내하고 사용자의 설비 선택을 기다리는 상태로 전환한다."""

    state["pending_action"] = PENDING_ACTION_SELECT_MOTOR
    state["pending_user_query"] = query
    state["pending_intent"] = supervisor_decision.primary_intent
    state["pending_supervisor_result"] = supervisor_decision
    state["selected_motor"] = None

    motors = get_equipment_repository().list_motors()
    append_assistant_message(state, _build_equipment_select_message(motors), detail=None)


def _handle_motor_selection(state: MutableMapping[str, Any], query: str, rag_agent: RagAgent) -> None:
    """pending_action == select_motor 상태에서 사용자의 설비 선택 입력을 처리한다."""

    repo = get_equipment_repository()
    motor = repo.find(query)

    if motor is None:
        motors = repo.list_motors()
        append_assistant_message(state, _build_equipment_reselect_message(motors), detail=None)
        return

    original_query = state.get("pending_user_query") or query

    state["selected_motor"] = motor
    state["equipment_id"] = motor.id
    state["pending_action"] = None
    state["pending_user_query"] = None
    state["pending_intent"] = None
    state["pending_supervisor_result"] = None

    result = run_pipeline(original_query, rag_agent, equipment_id=motor.id)
    state["last_user_query"] = original_query
    state["last_orchestration_result"] = result
    maintenance = result["maintenance"]
    if result["rag"] is not None:
        state["last_rag_result"] = result["rag"]
        if maintenance is not None and maintenance.checklist is not None:
            state["last_checklist"] = maintenance.checklist

    detail = dict(result)
    detail["is_follow_up"] = False
    detail["selected_motor"] = motor

    if (
        maintenance is not None
        and maintenance.checklist is not None
        and result["rag"] is not None
        and not result["rag"].insufficient_evidence
    ):
        answer = _build_scenario_selection_answer(motor, maintenance.checklist)
    else:
        answer = _build_chat_answer(detail)

    append_assistant_message(state, answer, detail=detail)

    state["show_work_order_quick_reply"] = bool(
        maintenance is not None and maintenance.checklist is not None and maintenance.work_order is None
    )


def handle_user_query(state: MutableMapping[str, Any], query: str, rag_agent: RagAgent) -> None:
    """사용자 질문/후속 요청 하나를 처리해 messages와 세션 상태를 갱신한다."""

    query = query.strip()
    if not query:
        return

    append_user_message(state, query)
    state["show_work_order_quick_reply"] = False

    if state.get("pending_action") == PENDING_ACTION_SELECT_MOTOR:
        _handle_motor_selection(state, query, rag_agent)
        return

    equipment_id = state.get("equipment_id") or DEFAULT_EQUIPMENT_ID

    if is_follow_up_work_order_request(query):
        last_rag_result = state.get("last_rag_result")
        last_checklist = state.get("last_checklist")
        if last_rag_result is None or last_checklist is None:
            append_assistant_message(state, NO_PRIOR_EVIDENCE_MESSAGE, detail=None)
            return

        detail = _build_follow_up_detail(query, equipment_id, last_rag_result, last_checklist)
        state["last_orchestration_result"] = detail
        append_assistant_message(state, _build_chat_answer(detail), detail=detail)
        return

    supervisor_decision = classify_request(query)
    if _is_scenario_request(supervisor_decision):
        _start_motor_selection(state, query, supervisor_decision)
        return

    result = run_pipeline(query, rag_agent, equipment_id=equipment_id)
    state["last_user_query"] = query
    state["last_orchestration_result"] = result
    maintenance = result["maintenance"]
    if result["rag"] is not None:
        state["last_rag_result"] = result["rag"]
        if maintenance is not None and maintenance.checklist is not None:
            state["last_checklist"] = maintenance.checklist

    detail = dict(result)
    detail["is_follow_up"] = False
    append_assistant_message(state, _build_chat_answer(detail), detail=detail)

    state["show_work_order_quick_reply"] = bool(
        maintenance is not None and maintenance.checklist is not None and maintenance.work_order is None
    )


def _build_work_order_text(detail: dict[str, Any]) -> str:
    maintenance: MaintenanceResult = detail["maintenance"]
    work_order = maintenance.work_order
    assert work_order is not None

    lines = [
        f"제목: {work_order.title}",
        f"설비 ID: {work_order.equipment_id}",
        f"접수 증상: {work_order.reported_symptom}",
        f"작업 목적: {work_order.purpose}",
        "점검 요청 항목:",
    ]
    lines.extend(f"  - {item}" for item in work_order.inspection_items)
    sources = ", ".join(work_order.reference_sources) if work_order.reference_sources else "없음"
    lines.append(f"참고 문서: {_display_document_name(sources)}")
    lines.append(f"상태: {work_order.status}")
    lines.append(f"안내: {work_order.disclaimer}")
    lines.append(f"안전 안내: {detail['safety_notice']}")
    return "\n".join(lines)


def _render_agent_process(detail: dict[str, Any], key_prefix: str) -> None:
    supervisor = detail.get("supervisor")
    maintenance = detail.get("maintenance")
    selected_motor: Motor | None = detail.get("selected_motor")

    if selected_motor is not None:
        st.markdown(f"**설비 정보** _(출처: {EQUIPMENT_JSON_PATH_LABEL} {EQUIPMENT_SOURCE_NOTE})_")
        st.write(f"- 설비명: {selected_motor.name}")
        st.write(f"- 설비 ID: {selected_motor.id}")
        st.write(f"- 구분명: {selected_motor.alias}")
        st.write(f"- 설치 위치: {selected_motor.location}")
        st.write(f"- 모델: {selected_motor.model}")
        st.write(f"- 현재 상태: {selected_motor.status}")
        st.divider()

    if detail.get("is_follow_up"):
        st.info(
            "후속 요청으로 판단해 별도 Supervisor 분류 없이 직전 RAG 근거와 "
            "점검 체크리스트를 재사용했습니다."
        )
    elif supervisor is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**범위 여부**:", "지원 범위 안" if supervisor.in_scope else "지원 범위 밖")
            st.write("**주요 의도**:", supervisor.primary_intent)
        with col2:
            outputs = ", ".join(supervisor.requested_outputs) if supervisor.requested_outputs else "없음"
            agents = ", ".join(supervisor.agents_to_call) if supervisor.agents_to_call else "없음"
            st.write("**요청 결과**:", outputs)
            st.write("**호출 Agent**:", agents)
        st.write("**판단 이유**:", supervisor.reason)

    if maintenance is None:
        return

    st.divider()
    if maintenance.checklist is not None:
        st.markdown(f"**점검 체크리스트** _(정비 근거 출처: {MANUAL_SOURCE_LABEL})_")
        for item in maintenance.checklist.items:
            document_label = _display_document_name(item.source_document) if item.source_document else ""
            source = f" _(출처: {document_label} p.{item.source_page})_" if item.source_page else ""
            st.write(f"- [{item.category}] {item.description}{source}")

    if maintenance.work_order is not None:
        st.markdown("**작업지시서 초안**")
        work_order = maintenance.work_order
        st.write(f"- 제목: {work_order.title}")
        st.write(f"- 설비 ID: {work_order.equipment_id}")
        st.write(f"- 접수 증상: {work_order.reported_symptom}")
        st.write(f"- 작업 목적: {work_order.purpose}")
        st.write("- 점검 요청 항목:")
        for item in work_order.inspection_items:
            st.write(f"  - {item}")
        sources = ", ".join(work_order.reference_sources) if work_order.reference_sources else "없음"
        st.write(f"- 참고 문서: {_display_document_name(sources)}")
        st.write(f"- 상태: {work_order.status}")
        st.caption(work_order.disclaimer)
        st.download_button(
            "작업지시서 다운로드 (.txt)",
            data=_build_work_order_text(detail),
            file_name=f"work_order_{detail['equipment_id']}.txt",
            mime="text/plain",
            key=f"{key_prefix}_download",
        )

    st.caption(f"담당자 검토 상태 안내: {maintenance.review_required_notice}")


def _render_rag_evidence(detail: dict[str, Any]) -> None:
    rag = detail.get("rag")
    if rag is None:
        st.info(
            "Supervisor 판단 결과 지원 범위를 벗어나 RAG Agent가 호출되지 않았습니다. "
            "이 프로젝트는 효성 저압전동기 매뉴얼 기반 질의만 지원합니다."
        )
        return

    if rag.insufficient_evidence:
        st.warning("근거 부족: 매뉴얼에서 충분한 근거를 찾지 못했습니다 (insufficient_evidence=True).")
    else:
        st.success("근거 충분: 매뉴얼에서 관련 근거를 찾았습니다.")

    for search_result in rag.results:
        with st.container(border=True):
            st.markdown(f"**검색 결과 {search_result.rank}**")
            st.write(f"- 문서명: {_display_document_name(search_result.document_name)}")
            st.write(f"- PDF 페이지: {search_result.page_number}")
            st.write(f"- 의미 유사도: {search_result.semantic_score:.4f}")
            st.write(f"- 키워드 보너스: {search_result.keyword_bonus:.4f}")
            st.write(f"- 최종 점수: {search_result.final_score:.4f}")
            core_evidence = extract_core_evidence(rag.query, search_result.page_text)
            st.write(f"- 핵심 근거: {core_evidence}")
            st.caption("검색에 사용된 청크")
            st.write(search_result.text)


def _render_pdf_source(detail: dict[str, Any]) -> None:
    rag = detail.get("rag")
    if rag is None or not rag.results:
        st.info("표시할 PDF 원문이 없습니다.")
        return

    st.caption(DISPLAY_NOTICE)
    for search_result in rag.results:
        st.markdown(
            f"**{_display_document_name(search_result.document_name)} p.{search_result.page_number}**"
        )
        formatted_page_text = format_page_text_for_display(search_result.page_text)
        safe_page_text = html.escape(formatted_page_text)
        st.markdown(
            f'<div class="pdf-page-viewer">{safe_page_text}</div>',
            unsafe_allow_html=True,
        )


def _render_message(message: dict[str, Any], key_prefix: str) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        detail = message.get("detail")
        if detail is not None:
            # 상세 기술 정보는 기본 대화보다 덜 강조되도록 하나의 접힌(compact) expander 안에
            # 탭으로 묶어서 배치한다. st.expander는 중첩할 수 없으므로 st.tabs로 구분한다.
            with st.expander("상세 근거 및 처리 과정 보기", type="compact"):
                process_tab, evidence_tab, source_tab = st.tabs(
                    ["Agent 처리 과정", "검색 근거 및 점수", "PDF 원문"]
                )
                with process_tab:
                    _render_agent_process(detail, key_prefix)
                with evidence_tab:
                    _render_rag_evidence(detail)
                with source_tab:
                    _render_pdf_source(detail)


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")

    st.markdown(
        """
        <style>
        .pdf-page-viewer {
            height: 500px;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 16px 18px;
            background-color: #f8fafc;
            color: #172033;
            border: 1px solid #d7dee7;
            border-radius: 8px;
            white-space: pre-wrap;
            word-break: keep-all;
            overflow-wrap: anywhere;
            font-family: "Malgun Gothic", "Apple SD Gothic Neo", sans-serif;
            font-size: 15px;
            line-height: 1.75;
        }

        /* 사용자/Assistant 말풍선을 시각적으로 구분한다. */
        div[data-testid="stChatMessageContent"] {
            width: fit-content;
        }
        div[data-testid="stChatMessage"]:has(div[aria-label^="Chat message from user"]) {
            flex-direction: row-reverse;
            justify-content: flex-end;
        }
        div[data-testid="stChatMessageContent"][aria-label^="Chat message from user"] {
            background-color: #2563eb;
            color: #ffffff;
            border-radius: 18px;
            padding: 12px 18px;
            max-width: 70%;
            line-height: 1.6;
        }
        div[data-testid="stChatMessageContent"][aria-label^="Chat message from user"] * {
            color: #ffffff;
        }
        div[data-testid="stChatMessageContent"][aria-label^="Chat message from assistant"] {
            background-color: #e8f0fe;
            color: #1f2933;
            border-radius: 18px;
            padding: 14px 18px;
            max-width: 78%;
            line-height: 1.75;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    init_session_state(st.session_state)
    rag_agent = get_rag_agent()

    with st.sidebar:
        st.markdown("#### 선택된 설비")
        selected_motor: Motor | None = st.session_state.get("selected_motor")
        if selected_motor is None:
            st.caption("선택된 설비: 없음")
        else:
            st.markdown(
                f"- {selected_motor.alias}\n"
                f"- {selected_motor.name}\n"
                f"- {selected_motor.id}\n"
                f"- {selected_motor.location}"
            )
        st.caption(f"담당 설비 목록: {EQUIPMENT_JSON_PATH_LABEL} {EQUIPMENT_SOURCE_NOTE}")
        new_chat_clicked = st.button("새 대화", width="stretch")
        st.divider()
        st.caption("예시 질문")
        example_rag_clicked = st.button("RAG 예시 질문", width="stretch")
        example_scenario_clicked = st.button("시나리오 예시 질문", width="stretch")

    if new_chat_clicked:
        reset_conversation(st.session_state)
        st.rerun()

    st.title(PAGE_TITLE)
    st.caption(
        "Supervisor Agent가 요청 의도를 분석하고, RAG Agent가 효성 매뉴얼에서 근거를 검색한 뒤, "
        "Maintenance Agent가 점검표와 작업지시서 초안을 생성합니다."
    )

    pending_query: str | None = None
    if example_rag_clicked:
        pending_query = RAG_EXAMPLE_QUESTION
    elif example_scenario_clicked:
        pending_query = SCENARIO_EXAMPLE_QUESTION

    chat_query = st.chat_input("질문 또는 증상을 입력하세요")
    if chat_query:
        pending_query = chat_query

    if pending_query:
        handle_user_query(st.session_state, pending_query, rag_agent)

    for index, message in enumerate(st.session_state["messages"]):
        _render_message(message, key_prefix=f"msg_{index}")

    if st.session_state.get("pending_action") == PENDING_ACTION_SELECT_MOTOR:
        motors = get_equipment_repository().list_motors()
        quick_select_cols = st.columns(len(motors))
        for col, motor in zip(quick_select_cols, motors):
            with col:
                if st.button(f"{motor.number}번 {motor.alias}", key=f"quick_select_motor_{motor.number}"):
                    handle_user_query(st.session_state, f"{motor.number}번", rag_agent)
                    st.rerun()

    if st.session_state.get("show_work_order_quick_reply"):
        if st.button(QUICK_REPLY_WORK_ORDER_LABEL, key="quick_reply_work_order"):
            handle_user_query(st.session_state, QUICK_REPLY_WORK_ORDER_QUESTION, rag_agent)
            st.rerun()


if __name__ == "__main__":
    main()
