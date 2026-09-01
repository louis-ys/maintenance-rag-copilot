"""src.equipment_repository 단위 테스트.

JSON 로딩과 번호/구분명/설비 ID/설비명 조회만 검증한다. 임베딩이나
벡터 검색을 전혀 사용하지 않는 일반 JSON 조회 모듈이어야 한다.
"""

from __future__ import annotations

from src.equipment_repository import EquipmentRepository


def _make_repository() -> EquipmentRepository:
    return EquipmentRepository()


def test_loads_json_successfully() -> None:
    repo = _make_repository()

    motors = repo.list_motors()

    assert len(motors) > 0


def test_list_motors_returns_three_motors() -> None:
    repo = _make_repository()

    motors = repo.list_motors()

    assert len(motors) == 3


def test_find_by_plain_number() -> None:
    repo = _make_repository()

    motor = repo.find("3")

    assert motor is not None
    assert motor.alias == "C모터"
    assert motor.id == "M-103"


def test_find_by_number_with_suffix() -> None:
    repo = _make_repository()

    motor = repo.find("3번")

    assert motor is not None
    assert motor.alias == "C모터"


def test_find_by_alias() -> None:
    repo = _make_repository()

    motor = repo.find("C모터")

    assert motor is not None
    assert motor.id == "M-103"


def test_find_by_equipment_id() -> None:
    repo = _make_repository()

    motor = repo.find("M-103")

    assert motor is not None
    assert motor.alias == "C모터"


def test_find_by_equipment_name() -> None:
    repo = _make_repository()

    motor = repo.find("컨베이어 구동 모터")

    assert motor is not None
    assert motor.alias == "C모터"
    assert motor.id == "M-103"


def test_find_with_invalid_number_returns_none() -> None:
    repo = _make_repository()

    motor = repo.find("7번")

    assert motor is None


def test_find_with_empty_input_returns_none() -> None:
    repo = _make_repository()

    assert repo.find("") is None
    assert repo.find(None) is None
