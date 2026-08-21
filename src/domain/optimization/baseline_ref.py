"""최적화의 출발점 (Anti-Corruption Layer VO).

Optimization Context 는 TrainingRun 도 ModelVersion 도 모른다.
알아야 하는 것은 "무엇을 최적화할 것이고, 그것이 원래 얼마나 좋았는가"뿐이다.

여기에도 **모듈 3 의 승인 통과 여부**가 함께 담긴다.
승인받지 않은 모델을 최적화하는 것은 순서가 틀린 일이기 때문이다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from domain.shared.errors import InvariantViolation


@dataclass(frozen=True, slots=True)
class BaselineModelRef:
    """최적화할 원본 모델."""

    model_version_id: str
    run_ref: str
    input_shape: tuple[int, ...]
    class_labels: tuple[str, ...]
    parameter_count: int
    mac_count: int
    accuracy: float
    macro_recall: float
    per_class_recall: Mapping[str, float] = field(default_factory=dict)
    accepted: bool = False
    """모듈 3 의 모델 승인 판정을 통과했는가."""

    def __post_init__(self) -> None:
        if not self.model_version_id.strip():
            raise InvariantViolation(
                "어느 모델을 최적화하는지 없으면 결과를 되돌릴 수 없다.",
                subject="model_version_id",
            )
        if not self.input_shape:
            raise InvariantViolation("입력 모양이 없다.", subject="input_shape")
        if len(self.class_labels) < 2:
            raise InvariantViolation("클래스가 둘 미만이다.", subject="class_labels")
        for name in ("parameter_count", "mac_count"):
            if getattr(self, name) < 0:
                raise InvariantViolation(f"{name} 는 음수일 수 없다.", subject=name)
        for name in ("accuracy", "macro_recall"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise InvariantViolation(f"{name} 는 0~1 이어야 한다.", subject=name)

    @property
    def missing_gates(self) -> tuple[str, ...]:
        return () if self.accepted else ("모듈 3 · 모델 승인 판정",)
