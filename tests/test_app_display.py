"""app.py 화면 표시 이름 변환 단위 테스트."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import DOCUMENT_DISPLAY_NAME, _display_document_name


def test_document_display_name_is_hyosung_manual_title() -> None:
    assert DOCUMENT_DISPLAY_NAME == "효성 저압전동기 취급설명서"


def test_display_document_name_replaces_internal_file_name() -> None:
    text = "hyosung_motor_manual_ko.pdf p.6"

    assert _display_document_name(text) == f"{DOCUMENT_DISPLAY_NAME} p.6"


def test_display_document_name_leaves_other_text_unchanged() -> None:
    text = "관련 없는 문자열"

    assert _display_document_name(text) == text
