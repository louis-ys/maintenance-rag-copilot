"""하이브리드 검색에서 사용하는 한국어 키워드 확장 규칙.

의미 검색(임베딩)만으로는 매뉴얼에서 쓰이는 동의어/관련어(예: '보충'과
'주입')를 완전히 포착하지 못할 수 있다. 질의에 포함된 대표 키워드를
관련어로 확장한 뒤, 확장된 단어가 실제로 등장하는 청크에 소폭의 가산점을
주어 재정렬하기 위한 규칙과 계산 함수를 제공한다.
"""

from __future__ import annotations

KEYWORD_EXPANSION_RULES: dict[str, list[str]] = {
    "보충": ["주입", "급유", "윤활"],
    "그리스": ["윤활", "급유"],
    "과열": ["온도 상승", "발열", "냉각", "통풍"],
    "소음": ["이음", "이상음"],
    "진동": ["흔들림"],
    "정렬": ["얼라인먼트", "축 정렬"],
    "기동": ["시동", "운전 시작"],
}

PER_TERM_BONUS = 0.03
MAX_KEYWORD_BONUS = 0.15


def expand_query_terms(query: str) -> set[str]:
    """질의에 등장한 키워드 자신과 그 확장어를 모두 모은 집합을 반환한다."""

    expanded_terms: set[str] = set()
    for keyword, synonyms in KEYWORD_EXPANSION_RULES.items():
        if keyword in query:
            expanded_terms.add(keyword)
            expanded_terms.update(synonyms)
    return expanded_terms


def compute_keyword_bonus(query: str, text: str) -> float:
    """확장된 키워드가 본문에 등장한 개수에 비례한 가산점을 계산한다.

    가산점은 MAX_KEYWORD_BONUS로 상한을 두어 의미 검색 점수를
    과도하게 압도하지 않도록 한다.
    """

    expanded_terms = expand_query_terms(query)
    if not expanded_terms:
        return 0.0

    matched_count = sum(1 for term in expanded_terms if term in text)
    return min(MAX_KEYWORD_BONUS, matched_count * PER_TERM_BONUS)
