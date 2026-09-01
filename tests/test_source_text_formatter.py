"""src.source_text_formatter 단위 테스트."""

from __future__ import annotations

from src.source_text_formatter import (
    DISPLAY_NOTICE,
    DISPLAY_TEXT_REPLACEMENTS,
    format_page_text_for_display,
)


def test_numbered_procedure_items_are_on_separate_lines() -> None:
    text = (
        "점검사항 그리스 주입형 베어링 1 운전상태에서 그리스를 보충하고 점검한다(적정량 확인) "
        "2 C3급 베어링인 경우 30분 이상 부하 운전한 후 점검한다. "
        "3 인버터 운전인 경우 전원을 직결로 투입하여 운전한 후 점검한다. "
        "4 운전중 전원을 차단하고 전원없이 회전이 유지되는 동안 점검한다. "
        "5 부하기기와 연결된 커플링을 분해하고 무부하 운전하여 점검한다."
    )

    formatted = format_page_text_for_display(text)
    lines = formatted.split("\n")

    for marker in ("1", "2", "3", "4", "5"):
        matching_lines = [line for line in lines if line.startswith(f"{marker} ")]
        assert matching_lines, f"번호 {marker}가 별도 줄로 시작하지 않았습니다: {formatted!r}"

    assert len({line for line in lines if line and line[0] in "12345" and line[1:2] == " "}) == 5


def test_symbols_are_shown_on_their_own_lines() -> None:
    text = "점검한다. ☞ 소음이 제거되면 정상 현상입니다. ※ 추가로 주입하면 됩니다."

    formatted = format_page_text_for_display(text)
    lines = formatted.split("\n")

    assert "☞" in lines
    assert "※" in lines


def test_korean_paren_markers_are_on_separate_lines() -> None:
    text = "점검 항목 (가) 축은 손을 돌렸을 때 부드럽게 회전하는가? (나) 윤활유는 적정한가?"

    formatted = format_page_text_for_display(text)
    lines = formatted.split("\n")

    assert any(line.startswith("(가)") for line in lines)
    assert any(line.startswith("(나)") for line in lines)


def test_section_heading_starts_new_line() -> None:
    text = "이전 문단 내용입니다. 3.4.3 시운전 후 주의사항 반복적인 시험기동은 위험합니다."

    formatted = format_page_text_for_display(text)
    lines = formatted.split("\n")

    assert any(line.startswith("3.4.3") for line in lines)


def test_known_word_split_errors_are_fixed() -> None:
    for wrong, right in DISPLAY_TEXT_REPLACEMENTS:
        formatted = format_page_text_for_display(f"문장 시작 {wrong} 문장 끝")
        assert wrong not in formatted
        assert right in formatted


def test_missing_tilde_in_time_range_is_fixed() -> None:
    text = "그리스를 보충한 후 2030분 이상 부하 운전한 후 점검한다."

    formatted = format_page_text_for_display(text)

    assert "2030분" not in formatted
    assert "20~30분" in formatted


def test_repeated_table_header_noise_is_collapsed() -> None:
    text = (
        "순순 순순 서서 서서 점검사항 점검사항 점검사항 점검사항 "
        "그리스 그리스 그리스 그리스 주입형 주입형 주입형 주입형 "
        "베어링 베어링 베어링 베어링 밀봉형 밀봉형 밀봉형 밀봉형"
    )

    formatted = format_page_text_for_display(text)

    assert formatted == "베어링 이음 점검 절차"


def test_normal_spacing_between_words_is_preserved() -> None:
    text = "베어링 온도와 진동이 일시적으로 증가할 수 있습니다."

    formatted = format_page_text_for_display(text)

    assert "베어링 온도와 진동이 일시적으로 증가할 수 있습니다." in formatted


def test_display_notice_mentions_only_whitespace_cleanup() -> None:
    assert "공백" in DISPLAY_NOTICE
    assert "의미" in DISPLAY_NOTICE
