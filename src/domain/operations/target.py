"""배포 대상. (실습 5-1)

한 대에 올리는 것과 수천 대에 올리는 것은 다른 일이다.
그 차이를 타입으로 남겨 둔다 — 모듈 6 의 OTA 가 이 자리에서 확장된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from domain.shared.errors import InvariantViolation


class TargetKind(Enum):
    DEVICE = "DEVICE"
    """한 대. 실습용이자 파일럿용."""

    DEVICE_GROUP = "DEVICE_GROUP"
    """같은 조건의 몇 대. 단계적 배포의 첫 칸."""

    FLEET = "FLEET"
    """전부. 여기에 바로 올리는 것을 실습 5-9 가 막는다."""


@dataclass(frozen=True, slots=True)
class DeploymentTarget:
    """어디에 올리는가."""

    kind: TargetKind
    identifier: str
    name: str = ""
    device_count: int = 1

    def __post_init__(self) -> None:
        if not self.identifier.strip():
            raise InvariantViolation(
                "어디에 올리는지 없으면 배포가 아니다.", subject="identifier"
            )
        if self.device_count < 1:
            raise InvariantViolation(
                "대상이 0대인 배포는 없다.", subject="device_count"
            )
        if self.kind is TargetKind.DEVICE and self.device_count != 1:
            raise InvariantViolation(
                "DEVICE 는 한 대다. 여러 대면 DEVICE_GROUP 이다.",
                subject="device_count",
            )

    @property
    def is_wide(self) -> bool:
        """되돌리기 어려운 규모인가."""
        return self.kind is TargetKind.FLEET or self.device_count > 10

    def describe(self) -> str:
        name = self.name or self.identifier
        return f"{name} [{self.kind.value} · {self.device_count}대]"
