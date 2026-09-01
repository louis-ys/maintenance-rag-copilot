"""JSON 기반 설비(모터) 조회 저장소.

시연용 가상 설비 데이터(`data/equipment/motors.json`)를 로딩해 번호/구분명/
설비 ID/설비명으로 조회하는 기능만 제공한다. 임베딩이나 벡터 검색은
사용하지 않고, 일반 JSON 조회(사전 매칭)로만 동작한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_EQUIPMENT_JSON_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "equipment" / "motors.json"
)


@dataclass(frozen=True)
class Motor:
    """시연용 가상 설비(모터) 한 대의 정보."""

    number: int
    alias: str
    id: str
    name: str
    location: str
    model: str
    status: str


class EquipmentRepository:
    """`data/equipment/motors.json`을 로딩해 설비 목록 조회를 제공한다."""

    def __init__(self, json_path: Path | str = DEFAULT_EQUIPMENT_JSON_PATH) -> None:
        self._json_path = Path(json_path)
        self._motors = self._load(self._json_path)

    @staticmethod
    def _load(json_path: Path) -> list[Motor]:
        if not json_path.exists():
            raise FileNotFoundError(f"설비 JSON 파일이 없습니다: {json_path}")

        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return [
            Motor(
                number=item["number"],
                alias=item["alias"],
                id=item["id"],
                name=item["name"],
                location=item["location"],
                model=item["model"],
                status=item["status"],
            )
            for item in data.get("motors", [])
        ]

    def list_motors(self) -> list[Motor]:
        """등록된 전체 설비 목록을 번호 순서 그대로 반환한다."""

        return list(self._motors)

    def find(self, user_input: str | None) -> Motor | None:
        """번호/"N번"/구분명/설비 ID/설비명 중 하나로 설비를 조회한다.

        일치하는 설비가 없으면 None을 반환한다.
        """

        if not user_input:
            return None

        text = user_input.strip()
        if not text:
            return None

        normalized = text[:-1].strip() if text.endswith("번") else text

        for motor in self._motors:
            candidates = (str(motor.number), motor.alias, motor.id, motor.name)
            if text in candidates or normalized in candidates:
                return motor

        return None
