"""결정적 템플릿 기반 Maintenance Agent.

RAG Agent가 반환한 검색 근거(SearchResponse)만을 재료로 점검
체크리스트와 작업지시서 초안을 만든다. 외부 LLM을 호출하지 않으며,
매뉴얼에 없는 고장 원인을 단정하거나 특정 부품 교체를 지시하지 않는다.
근거가 부족(insufficient_evidence=True)하면 어떤 문서도 만들지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.models import SearchResponse, SearchResult
from src.supervisor_agent import OUTPUT_MAINTENANCE_CHECKLIST, OUTPUT_WORK_ORDER

REVIEW_NOTICE = "본 결과는 AI가 생성한 초안이며, 담당자의 최종 검토와 확인이 필요합니다."
SAFETY_POLICY_CATEGORY = "시스템 안전정책"
MANUAL_EVIDENCE_CATEGORY = "매뉴얼 근거"

SAFETY_POLICY_ITEMS: tuple[str, ...] = (
    "설비 정지",
    "전원 차단",
    "담당자 안전절차 확인",
)

TOP_EVIDENCE_WINDOW = 3
EVIDENCE_MATCH_THRESHOLD = 2

# 질문/검색 근거가 서로 관련 있는지 판단하기 위한 주제 그룹과 대표 키워드.
QUESTION_KEYWORD_GROUPS: dict[str, tuple[str, ...]] = {
    "진동_소음": ("진동", "소음", "이음", "이상음", "흔들림"),
    "베어링_윤활": ("베어링", "그리스", "윤활", "급유", "주입"),
    "정렬": ("정렬", "얼라인먼트", "커플링", "축"),
    "과열_냉각": ("과열", "발열", "온도", "냉각", "통풍"),
    "전기": ("전압", "전류", "전원", "결선"),
    "부하": ("부하", "과부하", "정격"),
    "기동": ("기동", "시동"),
}

# 최종 안내 문구에 사용할 주제 그룹의 한국어 표시명.
TOPIC_LABELS: dict[str, str] = {
    "진동_소음": "진동·소음",
    "베어링_윤활": "베어링·윤활",
    "정렬": "정렬",
    "과열_냉각": "과열·냉각",
    "전기": "전기",
    "부하": "부하",
    "기동": "기동",
}

# 진동·소음 질문에서는 근거가 검색되어도 기본적으로 제외할 그룹.
# 허용 그룹은 진동_소음/베어링_윤활/정렬 세 개뿐이므로, 나머지 그룹은
# 검색 원문에 관련 키워드가 있어도 제외한다.
VIBRATION_NOISE_GROUP = "진동_소음"
VIBRATION_NOISE_EXCLUDED_GROUPS: tuple[str, ...] = ("과열_냉각", "부하", "전기", "기동")

# (그룹, 트리거 키워드, 체크리스트 항목 설명). 해당 그룹이 이번 질문에서
# 허용되고(_detect_allowed_groups), 검색된 근거 문장에 트리거 키워드가
# 하나라도 등장할 때만 항목을 생성한다. "확인 필요"류 표현만 사용하고
# 특정 부품 교체나 고장 원인을 단정하지 않는다.
MANUAL_CHECKLIST_RULES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("베어링_윤활", ("베어링",), "베어링 이상음 여부 확인 필요"),
    ("베어링_윤활", ("그리스", "윤활", "급유", "주입"), "베어링 그리스 상태 확인 필요"),
    ("정렬", ("정렬", "얼라인먼트", "커플링", "축"), "부하기기 연결 및 얼라인먼트 확인 필요"),
    ("과열_냉각", ("냉각", "통풍", "과열", "발열", "온도"), "통풍구와 냉각 상태 확인 필요"),
    ("부하", ("부하", "과부하", "정격"), "부하 조건이 정격 범위인지 확인 필요"),
    ("전기", ("전압", "전류", "전원", "결선"), "전원 및 결선 상태 확인 필요"),
    ("진동_소음", ("소음", "이음", "이상음", "진동", "흔들림"), "비정상 소음과 진동 발생 위치 확인 필요"),
    ("기동", ("기동", "시동"), "기동 시 이상 여부 확인 필요"),
)


@dataclass(frozen=True)
class ChecklistItem:
    """점검 체크리스트 한 항목."""

    category: str
    description: str
    source_document: str | None
    source_page: int | None


@dataclass(frozen=True)
class MaintenanceChecklist:
    """안전정책 항목과 매뉴얼 근거 항목으로 구성된 점검 체크리스트."""

    items: list[ChecklistItem]
    review_notice: str
    matched_topics: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkOrderDraft:
    """정비 작업지시서 초안."""

    title: str
    equipment_id: str
    reported_symptom: str
    purpose: str
    inspection_items: list[str]
    reference_sources: list[str]
    status: str
    disclaimer: str


@dataclass(frozen=True)
class MaintenanceResult:
    """Maintenance Agent의 최종 산출물."""

    checklist: MaintenanceChecklist | None
    work_order: WorkOrderDraft | None
    review_required_notice: str


def generate_maintenance_outputs(
    query: str,
    equipment_id: str,
    rag_response: SearchResponse,
    requested_outputs: list[str],
) -> MaintenanceResult | None:
    """근거가 충분할 때만 요청된 산출물(체크리스트/작업지시서)을 생성한다."""

    if rag_response.insufficient_evidence:
        return None

    checklist = build_checklist(rag_response)
    work_order = (
        build_work_order(query, equipment_id, checklist)
        if OUTPUT_WORK_ORDER in requested_outputs
        else None
    )

    return MaintenanceResult(
        checklist=checklist if OUTPUT_MAINTENANCE_CHECKLIST in requested_outputs else None,
        work_order=work_order,
        review_required_notice=REVIEW_NOTICE,
    )


def build_checklist(rag_response: SearchResponse) -> MaintenanceChecklist:
    """안전정책 항목과, 질문과 관련성이 확인된 검색 근거 항목만 결합한 체크리스트를 만든다."""

    items: list[ChecklistItem] = [
        ChecklistItem(
            category=SAFETY_POLICY_CATEGORY,
            description=description,
            source_document=None,
            source_page=None,
        )
        for description in SAFETY_POLICY_ITEMS
    ]

    allowed_groups = _detect_allowed_groups(rag_response.query, rag_response.results)

    seen_descriptions: set[str] = set()
    matched_groups: list[str] = []
    for result in rag_response.results:
        for group, trigger_keywords, description in MANUAL_CHECKLIST_RULES:
            if description in seen_descriptions:
                continue
            if group not in allowed_groups:
                continue
            if any(keyword in result.text for keyword in trigger_keywords):
                items.append(
                    ChecklistItem(
                        category=MANUAL_EVIDENCE_CATEGORY,
                        description=description,
                        source_document=result.document_name,
                        source_page=result.page_number,
                    )
                )
                seen_descriptions.add(description)
                if group not in matched_groups:
                    matched_groups.append(group)

    matched_topics = tuple(TOPIC_LABELS[group] for group in matched_groups)
    return MaintenanceChecklist(items=items, review_notice=REVIEW_NOTICE, matched_topics=matched_topics)


def _detect_query_groups(query: str) -> set[str]:
    """사용자 질문 문장에 직접 등장하는 주제 그룹을 찾는다."""

    return {
        group
        for group, keywords in QUESTION_KEYWORD_GROUPS.items()
        if any(keyword in query for keyword in keywords)
    }


def _detect_evidence_groups(results: list[SearchResult]) -> set[str]:
    """검색 결과 상위 N개 중 2개 이상에서 관련 내용이 확인되는 주제 그룹을 찾는다."""

    top_results = results[:TOP_EVIDENCE_WINDOW]
    evidence_groups: set[str] = set()
    for group, keywords in QUESTION_KEYWORD_GROUPS.items():
        matched_count = sum(
            1 for result in top_results if any(keyword in result.text for keyword in keywords)
        )
        if matched_count >= EVIDENCE_MATCH_THRESHOLD:
            evidence_groups.add(group)
    return evidence_groups


def _detect_allowed_groups(query: str, results: list[SearchResult]) -> set[str]:
    """질문 키워드 또는 상위 검색 근거로 뒷받침되는 주제 그룹만 허용한다.

    진동·소음 질문에서는 과열·냉각/부하/전기 그룹을 근거가 있어도
    기본적으로 제외해, 증상과 무관한 체크리스트 항목이 섞이지 않게 한다.
    """

    query_groups = _detect_query_groups(query)
    evidence_groups = _detect_evidence_groups(results)
    allowed_groups = query_groups | evidence_groups

    if VIBRATION_NOISE_GROUP in query_groups:
        excluded = set(VIBRATION_NOISE_EXCLUDED_GROUPS) - query_groups
        allowed_groups -= excluded

    return allowed_groups


def build_work_order(
    query: str, equipment_id: str, checklist: MaintenanceChecklist
) -> WorkOrderDraft:
    """체크리스트의 매뉴얼 근거 항목을 바탕으로 작업지시서 초안을 만든다."""

    manual_items = [item for item in checklist.items if item.category == MANUAL_EVIDENCE_CATEGORY]
    inspection_items = [item.description for item in manual_items]

    reference_sources: list[str] = []
    seen_sources: set[str] = set()
    for item in manual_items:
        source = f"{item.source_document} p.{item.source_page}"
        if source not in seen_sources:
            reference_sources.append(source)
            seen_sources.add(source)

    return WorkOrderDraft(
        title=f"설비 {equipment_id} 정비 작업지시서(초안)",
        equipment_id=equipment_id,
        reported_symptom=query,
        purpose="접수된 증상의 원인 가능성을 매뉴얼 근거로 점검하고 필요한 조치 여부를 판단하기 위함",
        inspection_items=inspection_items,
        reference_sources=reference_sources,
        status="담당자 검토 대기",
        disclaimer=f"AI 업무지원 초안입니다. {REVIEW_NOTICE}",
    )
