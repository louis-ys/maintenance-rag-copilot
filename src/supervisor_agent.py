"""규칙 기반 Supervisor Agent.

사용자 질의가 이 프로젝트(효성 저압전동기 유지보수)의 업무 범위에
해당하는지, 어떤 의도(단순 매뉴얼 질의 vs 정비 문서 요청)인지,
그리고 어떤 산출물과 Agent가 필요한지를 키워드 규칙으로 판단한다.

특정 문장을 통째로 하드코딩하지 않고, 모터 업무 키워드 / 점검표 키워드 /
작업지시서 키워드의 등장 여부만으로 분류한다.
"""

from __future__ import annotations

from dataclasses import dataclass

RAG_AGENT_NAME = "rag_agent"
MAINTENANCE_AGENT_NAME = "maintenance_agent"

INTENT_MANUAL_QUERY = "manual_query"
INTENT_MAINTENANCE_REQUEST = "maintenance_request"
INTENT_OUT_OF_SCOPE = "out_of_scope"

OUTPUT_MANUAL_ANSWER = "manual_answer"
OUTPUT_MAINTENANCE_CHECKLIST = "maintenance_checklist"
OUTPUT_WORK_ORDER = "work_order"

MOTOR_SCOPE_KEYWORDS: tuple[str, ...] = (
    "모터", "전동기", "베어링", "그리스", "윤활", "진동", "소음", "이음",
    "과열", "온도", "축", "정렬", "얼라인먼트", "냉각", "통풍", "전압",
    "전류", "기동", "시동", "결선", "회전", "부하",
)

CHECKLIST_KEYWORDS: tuple[str, ...] = (
    "점검표", "점검 표", "체크리스트", "체크 리스트",
    "점검항목", "점검 항목", "점검리스트", "점검 리스트",
)

WORK_ORDER_KEYWORDS: tuple[str, ...] = (
    "작업지시서", "작업 지시서", "작업지시", "작업 지시", "지시서",
)


@dataclass(frozen=True)
class SupervisorDecision:
    """Supervisor Agent의 판단 결과."""

    in_scope: bool
    primary_intent: str
    requested_outputs: list[str]
    reason: str
    agents_to_call: list[str]


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def classify_request(query: str) -> SupervisorDecision:
    """질의를 분석해 업무 범위/의도/요청 산출물/호출 Agent를 결정한다."""

    if not _contains_any(query, MOTOR_SCOPE_KEYWORDS):
        return SupervisorDecision(
            in_scope=False,
            primary_intent=INTENT_OUT_OF_SCOPE,
            requested_outputs=[],
            reason="질문에 모터/전동기 유지보수 관련 키워드가 없어 지원 범위를 벗어납니다.",
            agents_to_call=[],
        )

    wants_checklist = _contains_any(query, CHECKLIST_KEYWORDS)
    wants_work_order = _contains_any(query, WORK_ORDER_KEYWORDS)

    if wants_checklist or wants_work_order:
        return _build_maintenance_decision(wants_checklist, wants_work_order)

    return SupervisorDecision(
        in_scope=True,
        primary_intent=INTENT_MANUAL_QUERY,
        requested_outputs=[OUTPUT_MANUAL_ANSWER],
        reason="모터 관련 일반 질문으로 판단되어 매뉴얼 검색 결과만 제공합니다.",
        agents_to_call=[RAG_AGENT_NAME],
    )


def _build_maintenance_decision(wants_checklist: bool, wants_work_order: bool) -> SupervisorDecision:
    requested_outputs: list[str] = []
    reason_parts: list[str] = []

    if wants_checklist:
        requested_outputs.append(OUTPUT_MAINTENANCE_CHECKLIST)
        reason_parts.append("점검표(체크리스트) 요청")
    if wants_work_order:
        requested_outputs.append(OUTPUT_WORK_ORDER)
        reason_parts.append("작업지시서 요청")

    reason = " 및 ".join(reason_parts) + "이 감지되어 매뉴얼 검색 후 정비 문서를 생성합니다."

    return SupervisorDecision(
        in_scope=True,
        primary_intent=INTENT_MAINTENANCE_REQUEST,
        requested_outputs=requested_outputs,
        reason=reason,
        agents_to_call=[RAG_AGENT_NAME, MAINTENANCE_AGENT_NAME],
    )
