"""Supervisor → RAG → Maintenance Agent 순서로 실행하는 오케스트레이터."""

from __future__ import annotations

from typing import Any

from src.maintenance_agent import MaintenanceResult, generate_maintenance_outputs
from src.models import SearchResponse, SearchResult
from src.rag_agent import RagAgent
from src.supervisor_agent import (
    OUTPUT_MAINTENANCE_CHECKLIST,
    OUTPUT_WORK_ORDER,
    SupervisorDecision,
    classify_request,
)

SAFETY_NOTICE = "본 시스템은 참고용 AI 도구이며, 실제 정비/안전 조치는 반드시 담당자가 최종 판단합니다."
DEFAULT_EQUIPMENT_ID = "M-101"


def run_pipeline(
    query: str,
    rag_agent: RagAgent,
    equipment_id: str = DEFAULT_EQUIPMENT_ID,
) -> dict[str, Any]:
    """Supervisor -> RAG -> Maintenance 순서로 파이프라인을 실행해 결과 dict를 반환한다."""

    supervisor_decision = classify_request(query)

    if not supervisor_decision.in_scope:
        return _build_result(
            query=query,
            equipment_id=equipment_id,
            supervisor=supervisor_decision,
            rag=None,
            maintenance=None,
            final_message=(
                "이 질문은 효성 저압전동기 유지보수 지원 범위를 벗어납니다. "
                "모터/전동기 관련 질문을 입력해 주세요."
            ),
        )

    rag_response = rag_agent.search(query)

    needs_maintenance = bool(
        set(supervisor_decision.requested_outputs)
        & {OUTPUT_MAINTENANCE_CHECKLIST, OUTPUT_WORK_ORDER}
    )

    maintenance_result: MaintenanceResult | None = None
    if not rag_response.insufficient_evidence and needs_maintenance:
        maintenance_result = generate_maintenance_outputs(
            query=query,
            equipment_id=equipment_id,
            rag_response=rag_response,
            requested_outputs=supervisor_decision.requested_outputs,
        )

    final_message = _build_final_message(rag_response, maintenance_result, needs_maintenance)

    return _build_result(
        query=query,
        equipment_id=equipment_id,
        supervisor=supervisor_decision,
        rag=rag_response,
        maintenance=maintenance_result,
        final_message=final_message,
    )


def _build_final_message(
    rag_response: SearchResponse,
    maintenance_result: MaintenanceResult | None,
    needs_maintenance: bool,
) -> str:
    if rag_response.insufficient_evidence:
        return "매뉴얼에서 충분한 근거를 찾지 못했습니다. 담당자의 추가 확인이 필요합니다."

    if needs_maintenance and maintenance_result is not None:
        evidence_source = _evidence_source_for_maintenance(rag_response, maintenance_result)
        documents_label, pages_label = evidence_source
        topic_prefix = _describe_topics(maintenance_result)
        result_types_label = _describe_result_types(maintenance_result)
        return (
            f"{documents_label} {pages_label}페이지의 검색 근거를 바탕으로 "
            f"{topic_prefix}{result_types_label}을 생성했습니다. "
            "최종 정비 판단과 승인은 설비보전 담당자가 수행해야 합니다."
        )

    documents_label, pages_label = _evidence_source_from_results(rag_response.results)
    return (
        f"{documents_label} {pages_label}페이지의 매뉴얼 검색 근거를 확인했습니다. "
        "최종 정비 판단은 설비보전 담당자가 수행해야 합니다."
    )


def _evidence_source_for_maintenance(
    rag_response: SearchResponse, maintenance_result: MaintenanceResult
) -> tuple[str, str]:
    """정비 문서 생성에 실제로 사용된 근거(체크리스트 매뉴얼 항목)를 기준으로 문서명/페이지를 구성한다.

    필터링으로 제외된 근거(예: 진동·소음 질문의 냉각/부하 항목)까지 안내에
    섞이지 않도록, 검색 결과 전체가 아니라 실제로 채택된 항목만 사용한다.
    """

    checklist = maintenance_result.checklist
    if checklist is None:
        return _evidence_source_from_results(rag_response.results)

    manual_items = [item for item in checklist.items if item.source_document is not None]
    if not manual_items:
        return _evidence_source_from_results(rag_response.results)

    documents = _unique_in_order(item.source_document for item in manual_items if item.source_document)
    pages = _unique_in_order(item.source_page for item in manual_items if item.source_page is not None)
    return ", ".join(documents), _format_page_ranges(pages)


def _evidence_source_from_results(results: list[SearchResult]) -> tuple[str, str]:
    documents = _unique_in_order(result.document_name for result in results)
    pages = _unique_in_order(result.page_number for result in results)
    return ", ".join(documents), _format_page_ranges(pages)


def _describe_topics(maintenance_result: MaintenanceResult) -> str:
    checklist = maintenance_result.checklist
    if checklist is None or not checklist.matched_topics:
        return ""
    return ", ".join(checklist.matched_topics) + " 관련 "


def _describe_result_types(maintenance_result: MaintenanceResult) -> str:
    parts: list[str] = []
    if maintenance_result.checklist is not None:
        parts.append("점검 체크리스트 초안")
    if maintenance_result.work_order is not None:
        parts.append("작업지시서 초안")
    return " 및 ".join(parts) if parts else "정비 문서 초안"


def _unique_in_order(values: Any) -> list[Any]:
    seen: list[Any] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def _format_page_ranges(pages: list[int]) -> str:
    """페이지 번호 목록을 연속 구간은 '6~7'처럼, 나머지는 쉼표로 묶어 표시한다."""

    if not pages:
        return ""

    sorted_pages = sorted(set(pages))
    ranges: list[str] = []
    start = prev = sorted_pages[0]
    for page in sorted_pages[1:]:
        if page == prev + 1:
            prev = page
            continue
        ranges.append(f"{start}~{prev}" if start != prev else f"{start}")
        start = prev = page
    ranges.append(f"{start}~{prev}" if start != prev else f"{start}")
    return ", ".join(ranges)


def _build_result(
    query: str,
    equipment_id: str,
    supervisor: SupervisorDecision,
    rag: SearchResponse | None,
    maintenance: MaintenanceResult | None,
    final_message: str,
) -> dict[str, Any]:
    return {
        "query": query,
        "equipment_id": equipment_id,
        "supervisor": supervisor,
        "rag": rag,
        "maintenance": maintenance,
        "final_message": final_message,
        "safety_notice": SAFETY_NOTICE,
    }
