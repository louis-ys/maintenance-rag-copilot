"""PDF 페이지 전체 텍스트를 화면 표시용으로 정리하는 포맷터.

RAG 검색/임베딩에 사용되는 텍스트(청크, 인덱스)에는 전혀 영향을 주지 않으며,
이 모듈의 함수는 오직 화면에 보여줄 문자열을 만드는 데에만 쓰인다.
"""

from __future__ import annotations

import re

DISPLAY_NOTICE = (
    "PDF 추출 내용에서 공백과 줄바꿈만 화면 표시용으로 정리했습니다. "
    "원문의 의미와 순서는 변경하지 않았습니다."
)

# 실제로 확인된 PDF 텍스트 추출 시 단어 내부에 잘못 삽입된 공백/붙어버린
# 단어만 보수적으로 수정한다. 확실하지 않은 패턴은 추가하지 않는다.
DISPLAY_TEXT_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("점 검", "점검"),
    ("원활하 게", "원활하게"),
    ("있 음", "있음"),
    ("다 시", "다시"),
    ("그리 스", "그리스"),
    ("상승되 면", "상승되면"),
    ("마십시 오", "마십시오"),
    ("즉시모터", "즉시 모터"),
    ("운전한후", "운전한 후"),
    ("전원없이", "전원 없이"),
    ("운전중", "운전 중"),
    ("시운전시", "시운전 시"),
    ("수있", "수 있"),
    ("할수", "할 수"),
    ("될수", "될 수"),
    ("2030분", "20~30분"),
    ("즉시 모 터", "즉시 모터"),
    ("있 습니다", "있습니다"),
    ("합 니다", "합니다"),
    ("사용하 십시오", "사용하십시오"),
    ("주 파수", "주파수"),
    ("전압강하가 크게 되므로, 가능한 배선 에", "전압강하가 크게 되므로, 가능한 배선에"),
)

# PDF 표 헤더가 같은 단어를 열마다 반복 추출한 노이즈(예: "점검사항 점검사항
# 점검사항 점검사항 그리스 그리스 그리스 그리스 ...")를 화면 표시용으로만
# 축약한다. 검색/임베딩용 텍스트에는 적용하지 않는다.
# 같은 단어가 2회 이상 연속되는 구간을 후보로 잡되, 실제 축약은 표 헤더로
# 보이는 단어(_TABLE_HEADER_NOISE_TRIGGER_WORDS)가 블록에 2개 이상 포함된
# 경우에만 수행하므로 일반 문장의 우연한 반복에는 영향을 주지 않는다.
_REPEATED_WORD_RUN_RE = re.compile(r"(\S+)(?:\s+\1){1,}")
_TABLE_HEADER_NOISE_TRIGGER_WORDS = frozenset({"점검사항", "그리스", "베어링", "주입형", "밀봉형"})
_TABLE_HEADER_NOISE_LABEL = "베어링 이음 점검 절차"

# "3.4.3", "4.2.1"처럼 절 번호로 보이는 섹션 제목 앞에 줄바꿈을 넣는다.
_SECTION_HEADING_RE = re.compile(r"(\d{1,2}\.\d{1,2}\.\d{1,2})(?=\s)")

# "(가)", "(나)"처럼 괄호 안에 한글 한 글자만 있는 절차 표시 앞에 줄바꿈을 넣는다.
_KOREAN_PAREN_MARKER_RE = re.compile(r"\(([가-힣])\)")

# ※, ☞, − 기호의 앞뒤를 문단으로 구분한다.
_SYMBOL_PARAGRAPH_RE = re.compile(r"\s*([※☞−])\s*")

# ") 2 ..." 처럼 마침표/괄호 뒤에 바로 오는 1~5 번호 절차를 새 줄로 옮긴다.
_NUMBERED_ITEM_AFTER_PAREN_RE = re.compile(r"(?<=\))\s+([1-5])(?=\s)")

# "...다. 3 ..." 처럼 마침표나 한글 뒤에 오고, 뒤이어 한글 단어가 나오는
# 1~5 번호 절차를 새 줄로 옮긴다. 소수점 표기(예: 1.5g)나 영문 뒤에 오는
# 숫자(예: 1 HOT)는 대상에서 제외된다.
_NUMBERED_ITEM_GENERAL_RE = re.compile(r"(?<=[.가-힣])\s+([1-5])(?=\s[가-힣])")

_TAB_OR_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_BLANK_LINE_RE = re.compile(r"\n{2,}")


def _apply_word_fixes(text: str) -> str:
    for wrong, right in DISPLAY_TEXT_REPLACEMENTS:
        text = text.replace(wrong, right)
    return text


def _collapse_table_header_noise(text: str) -> str:
    """같은 단어가 3회 이상 연속 반복되는 표 추출 노이즈 구간을 축약한다.

    인접한 반복 구간들(공백만 사이에 있는 경우)을 하나의 노이즈 블록으로 묶고,
    블록에 표 헤더로 보이는 단어가 2개 이상 포함된 경우에만 축약한다.
    """

    matches = list(_REPEATED_WORD_RUN_RE.finditer(text))
    if not matches:
        return text

    blocks: list[list[re.Match[str]]] = []
    for match in matches:
        if blocks and text[blocks[-1][-1].end():match.start()].strip() == "":
            blocks[-1].append(match)
        else:
            blocks.append([match])

    result = text
    for block in reversed(blocks):
        words = {m.group(1) for m in block}
        if len(words & _TABLE_HEADER_NOISE_TRIGGER_WORDS) >= 2:
            start, end = block[0].start(), block[-1].end()
            result = result[:start] + _TABLE_HEADER_NOISE_LABEL + result[end:]
    return result


def format_page_text_for_display(text: str) -> str:
    """페이지 전체 텍스트를 화면에 읽기 좋게 정리한다.

    개행/공백 정리와 보수적인 단어 분리 오류 치환만 수행하며,
    원문의 단어와 순서는 바꾸지 않는다. 검색/임베딩용 텍스트에는 사용하지 않는다.
    """

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _TAB_OR_MULTI_SPACE_RE.sub(" ", normalized)
    normalized = _apply_word_fixes(normalized)
    normalized = _collapse_table_header_noise(normalized)

    normalized = _SECTION_HEADING_RE.sub(r"\n\1", normalized)
    normalized = _KOREAN_PAREN_MARKER_RE.sub(lambda m: f"\n({m.group(1)})", normalized)
    normalized = _SYMBOL_PARAGRAPH_RE.sub(lambda m: f"\n{m.group(1)}\n", normalized)
    normalized = _NUMBERED_ITEM_AFTER_PAREN_RE.sub(r"\n\1", normalized)
    normalized = _NUMBERED_ITEM_GENERAL_RE.sub(r"\n\1", normalized)

    lines = [line.strip() for line in normalized.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)
