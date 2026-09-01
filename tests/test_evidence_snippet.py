"""src.evidence_snippet 단위 테스트.

실제 PDF나 임베딩 모델에 의존하지 않도록, 매뉴얼 페이지에서 발췌한 것과
같은 형태의 페이지 전체 텍스트를 직접 구성해 핵심 근거 선택 로직만 검증한다.
"""

from __future__ import annotations

from src.evidence_snippet import MAX_SNIPPET_LENGTH, extract_core_evidence

# 실제 매뉴얼 6페이지 내용을 축약 없이 그대로 옮긴 문단(시운전 전 확인 절차 일부).
VIBRATION_NOISE_PAGE_TEXT = (
    "3.4 시운전 3.4.1 시운전 전 확인 점검 항목 (가) 축은 손을 돌렸을 때 부드럽게 회전하는가? "
    "(나) 윤활유는 적정한가? (다) Bolt 누락 개소는 없는가? (라) 냉각수는 확보되어 있는가? "
    "만일 반복적인 시험기동을 한다면 과열을 방지하기 위해 시험기동 사이에 전동기가 냉각될 수 있도록 "
    "충분한 시간을 두어야 합니다. 무부하 상태에서 운전을 하면서 회전상태를 점검하고, 베어링의 이상음이 "
    "없는지 확인하여 주십시오. 만약 과도한 소음, 진동, 이상음(반복적인 찰깍찰깍 하는 소리 또는 치는 소리 "
    "등)이 나면 즉시 모터를 멈추고 당사 A/S부서나 특약점으로 연락하십시오. 운전 중 이상발열 현상이 발생하는 "
    "경우 온도상승 한계치를 초과하는지를 점검하여야 합니다."
)


def test_core_evidence_uses_page_text_not_only_chunk() -> None:
    chunk_text = "베어링의 이상음이 없는지 확인하여 주십시오."

    evidence = extract_core_evidence("모터에서 진동과 소음이 발생합니다.", VIBRATION_NOISE_PAGE_TEXT)

    assert len(evidence) > len(chunk_text)


def test_core_evidence_ends_with_a_complete_sentence() -> None:
    evidence = extract_core_evidence("모터에서 진동과 소음이 발생합니다.", VIBRATION_NOISE_PAGE_TEXT)

    stripped = evidence.rstrip()
    ends_properly = stripped[-1:] in ".!?" or stripped.endswith(("합니다", "하십시오", "바랍니다"))
    assert ends_properly, f"완결되지 않은 문장으로 끝났습니다: {evidence!r}"


def test_core_evidence_length_within_max() -> None:
    evidence = extract_core_evidence("모터에서 진동과 소음이 발생합니다.", VIBRATION_NOISE_PAGE_TEXT)

    assert len(evidence) <= MAX_SNIPPET_LENGTH


def test_core_evidence_prioritizes_vibration_and_noise_content() -> None:
    evidence = extract_core_evidence("모터에서 진동과 소음이 발생합니다.", VIBRATION_NOISE_PAGE_TEXT)

    assert "이상음" in evidence
    assert "멈추" in evidence


def test_core_evidence_excludes_trailing_incomplete_fragment() -> None:
    page_text_with_cut_ending = VIBRATION_NOISE_PAGE_TEXT + " 과부하 운전으로 인하여 소손 될 수 있습니다 . 명판에 표시된 정격 전류치 이상의 전류로 운전하지 마십시오. 과"

    evidence = extract_core_evidence("모터에서 진동과 소음이 발생합니다.", page_text_with_cut_ending)

    assert not evidence.rstrip().endswith("과")


def test_core_evidence_falls_back_to_neighbor_when_little_is_relevant() -> None:
    unrelated_page_text = (
        "2.4 설치 해발고도 1000m이하이고, 주위온도 -15℃∼40℃ 사이인 장소에서 사용하여 주십시오. "
        "설치 장소는 먼지와 습기가 적고 통풍이 잘 되는 곳이어야 합니다. 진동이 발생할 수 있는 장소는 "
        "피하여 설치하시기 바랍니다. 배수가 잘 되는 평탄한 기초 위에 견고하게 고정하여 주십시오."
    )

    evidence = extract_core_evidence("모터 설치 장소는 어떻게 선정하나요?", unrelated_page_text)

    assert evidence != ""


def test_core_evidence_excludes_unrelated_electromagnetic_switch_sentence() -> None:
    page_text = (
        VIBRATION_NOISE_PAGE_TEXT
        + " 1차측 전자개폐기가 소손된 경우 즉시 교체하고 절연 상태를 점검하십시오."
    )

    evidence = extract_core_evidence("모터에서 진동과 소음이 발생합니다.", page_text)

    assert "전자개폐기" not in evidence


def test_core_evidence_excludes_unrelated_long_term_storage_sentence() -> None:
    page_text = (
        VIBRATION_NOISE_PAGE_TEXT
        + " 장기간 보관된 전동기는 절연저항을 측정한 후 사용하십시오."
    )

    evidence = extract_core_evidence("모터에서 진동과 소음이 발생합니다.", page_text)

    assert "장기간 보관" not in evidence


def test_core_evidence_does_not_pad_short_relevant_content_with_unrelated_sentences() -> None:
    page_text = (
        "설치 장소는 먼지와 습기가 적어야 합니다. 배수가 잘 되는 평탄한 기초 위에 견고하게 고정하여 주십시오. "
        "베어링에서 이상음이 발생하면 즉시 점검하십시오. "
        "1차측 전자개폐기는 정기적으로 점검하십시오. 장기간 보관된 전동기는 절연저항을 측정하십시오."
    )

    evidence = extract_core_evidence("모터에서 진동과 소음이 발생합니다.", page_text)

    assert evidence == "베어링에서 이상음이 발생하면 즉시 점검하십시오."
