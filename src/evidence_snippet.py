"""검색 결과의 근거 페이지에서 질문과 관련된 핵심 근거만 뽑아 보여주기 위한 유틸.

페이지 전체 텍스트(page_text)를 문장 단위로 나눈 뒤, 페이지 경계에서 잘린
미완성 문장은 제외하고, 질문과 관련된 완결된 문장을 원문 순서 그대로 최대
3개, 최대 350자까지만 구성한다. 관련 문장이 1~2개뿐이면 길이를 채우기 위해
무관한 문장을 추가하지 않고 짧은 상태로 그대로 보여준다. 임베딩 검색이나
청킹에는 영향을 주지 않으며 화면 표시 전용이다.
"""

from __future__ import annotations

import re

from src.maintenance_agent import QUESTION_KEYWORD_GROUPS
from src.source_text_formatter import DISPLAY_TEXT_REPLACEMENTS

MAX_SNIPPET_LENGTH = 350
MAX_SNIPPET_SENTENCES = 3

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|(?<=다)\s+(?=[가-힣0-9])")

# 문장이 완결되었다고 볼 수 있는 어미. 마침표/물음표/느낌표로 끝나거나
# 이 어미들 중 하나로 끝나면 완결된 문장으로 판단한다.
_COMPLETE_ENDING_SUFFIXES = ("합니다", "하십시오", "바랍니다")

# PDF 표 추출 시 같은 단어가 3회 이상 연달아 반복되는 구간(예: 표 헤더가
# 열마다 중복 추출된 경우)은 실제 문장이 아니므로 핵심 근거 후보에서 제외한다.
_DUPLICATE_TOKEN_RUN_RE = re.compile(r"(\S+)(?:\s+\1){2,}")


def _is_garbled_fragment(sentence: str) -> bool:
    return bool(_DUPLICATE_TOKEN_RUN_RE.search(sentence))


# 진동·소음 질문에서는 베어링/그리스/윤활/커플링 근거도 함께 우선 노출한다.
# "축", "정렬"처럼 이 문서에서 다른 문맥(예: 시운전 전 점검항목)에도 흔히
# 등장하는 단어는 제외해 관련 없는 문장이 섞이지 않도록 한다.
_VIBRATION_NOISE_GROUP = "진동_소음"
_VIBRATION_NOISE_EXTRA_TERMS: tuple[str, ...] = ("베어링", "그리스", "윤활", "급유", "주입", "커플링")


def normalize_display_text(text: str) -> str:
    """실제로 확인된 PDF 단어 분리 오류만 보수적으로 정리한다.

    한글 음절 사이의 공백을 일괄적으로 제거하지 않으며, 정상적인 단어 사이
    공백은 그대로 보존한다.
    """

    normalized = text
    for wrong, right in DISPLAY_TEXT_REPLACEMENTS:
        normalized = normalized.replace(wrong, right)
    return normalized


def _split_sentences(text: str) -> list[str]:
    sentences = [sentence.strip() for sentence in _SENTENCE_SPLIT_RE.split(text) if sentence.strip()]
    if sentences:
        return sentences
    return [text.strip()] if text.strip() else []


def _is_complete_sentence(sentence: str) -> bool:
    stripped = sentence.strip()
    if not stripped:
        return False
    if stripped[-1] in ".!?":
        return True
    return stripped.endswith(_COMPLETE_ENDING_SUFFIXES)


def _usable_sentence_indices(sentences: list[str]) -> list[int]:
    """페이지 마지막/첫 문장이 잘린 것으로 보이거나, 표 헤더가 중복 추출된
    구간으로 보이면 후보에서 제외한다."""

    indices = list(range(len(sentences)))
    if len(indices) > 1 and not _is_complete_sentence(sentences[indices[-1]]):
        indices = indices[:-1]
    if len(indices) > 1 and not _is_complete_sentence(sentences[indices[0]]):
        indices = indices[1:]
    return [index for index in indices if not _is_garbled_fragment(sentences[index])]


def _extract_query_terms(query: str) -> list[str]:
    """질문 그룹 및 허용 보조 그룹의 키워드만 근거 문장 선택 기준으로 사용한다.

    질문 원문 단어를 그대로 매칭에 사용하면 "점검"처럼 무관한 문장에도
    흔히 등장하는 단어 때문에 관계없는 문장이 섞일 수 있어 사용하지 않는다.
    """

    matched_groups = [
        group
        for group, keywords in QUESTION_KEYWORD_GROUPS.items()
        if any(keyword in query for keyword in keywords)
    ]

    terms: list[str] = []
    for group in matched_groups:
        for keyword in QUESTION_KEYWORD_GROUPS[group]:
            if keyword not in terms:
                terms.append(keyword)

    if _VIBRATION_NOISE_GROUP in matched_groups:
        for keyword in _VIBRATION_NOISE_EXTRA_TERMS:
            if keyword not in terms:
                terms.append(keyword)

    return terms


def _rank_relevant_indices(sentences: list[str], usable_indices: list[int], query_terms: list[str]) -> list[int]:
    """일치하는 키워드 수가 많은 문장을 우선하고, 동점이면 원문 순서를 따른다."""

    scored = [
        (index, sum(1 for term in query_terms if term in sentences[index]))
        for index in usable_indices
    ]
    scored = [(index, score) for index, score in scored if score > 0]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [index for index, _ in scored]


def _join_sentences(sentences: list[str], indices: list[int]) -> str:
    ordered = sorted(indices)
    return " ".join(sentences[i] for i in ordered).strip()


def _select_within_budget(sentences: list[str], ranked_indices: list[int], max_count: int) -> list[int]:
    """관련도 순으로 문장을 최대 max_count개까지 채우되, 최대 길이(350자)를
    넘기면 더 추가하지 않는다. 길이를 채우기 위한 무관한 문장 추가는 하지 않는다."""

    selected: list[int] = []
    for index in ranked_indices:
        if len(selected) >= max_count:
            break
        candidate = sorted(selected + [index])
        candidate_text = _join_sentences(sentences, candidate)
        if len(candidate_text) <= MAX_SNIPPET_LENGTH or not selected:
            selected = candidate
    return selected


def extract_core_evidence(query: str, page_text: str) -> str:
    """페이지 전체 텍스트에서 질문과 관련된 완결된 문장만 골라 핵심 근거를 만든다.

    관련 문장을 원문 순서 그대로 최대 3개, 최대 350자까지만 담으며, 관련
    문장이 1~2개뿐이어도 길이를 채우기 위해 무관한 문장을 추가하지 않는다.
    """

    normalized = normalize_display_text(page_text)
    sentences = _split_sentences(normalized)
    if not sentences:
        return normalized[:MAX_SNIPPET_LENGTH]

    usable_indices = _usable_sentence_indices(sentences)
    if not usable_indices:
        usable_indices = list(range(len(sentences)))

    query_terms = _extract_query_terms(query)
    ranked_relevant = _rank_relevant_indices(sentences, usable_indices, query_terms)
    if not ranked_relevant:
        ranked_relevant = usable_indices

    selected = _select_within_budget(sentences, ranked_relevant, max_count=MAX_SNIPPET_SENTENCES)
    return _join_sentences(sentences, selected)
