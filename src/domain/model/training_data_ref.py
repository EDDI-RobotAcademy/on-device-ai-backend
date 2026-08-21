"""학습에 쓸 데이터에 대한 참조 (Anti-Corruption Layer VO).

Model Context 는 Dataset 도, QualityAssessment 도 모른다.
알아야 하는 것은 "어디에 있는 무엇을, 어떤 열로, 어떻게 잘라서 볼 것인가"뿐이다.

번역기는 Application Layer 에 있다. (application/model/training_data_mapper.py)

여기에는 **두 게이트를 통과했다는 증거**도 함께 담는다.
그 증거 없이 학습을 시작할 수 없게 하려는 것이다 — 모듈 1과 2가 존재한 이유다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from domain.shared.errors import InvariantViolation


@dataclass(frozen=True, slots=True)
class TrainingDataRef:
    """학습 데이터 한 덩어리."""

    dataset_ref: str
    uri: str
    source_format: str = "CSV"

    feature_fields: tuple[str, ...] = field(default_factory=tuple)
    label_field: str | None = None
    time_field: str | None = None

    normalization: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    """실습 1-7 에서 **train 분할에서만** 뽑은 (평균, 표준편차).

    학습과 배포에서 같은 값을 써야 한다. 그래서 데이터가 아니라 계약으로 들고 다닌다.
    """

    split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15)

    readiness_certified: bool = False
    """모듈 1의 학습 착수 판정을 통과했는가."""

    quality_gate_passed: bool = False
    """모듈 2의 Data Quality Gate 를 통과했는가."""

    def __post_init__(self) -> None:
        if not self.dataset_ref.strip():
            raise InvariantViolation(
                "어느 Dataset 으로 학습했는지 없으면 모델을 되돌릴 수 없다.",
                subject="dataset_ref",
            )
        if not self.uri.strip():
            raise InvariantViolation("학습 데이터 위치가 없다.", subject="uri")
        if not self.feature_fields:
            raise InvariantViolation("입력 필드가 없다.", subject="feature_fields")
        if not self.label_field:
            raise InvariantViolation(
                "라벨 필드가 없다. 지도학습을 할 수 없다.", subject="label_field"
            )
        total = sum(self.split_ratio)
        if abs(total - 1.0) > 1e-9:
            raise InvariantViolation(
                f"분할 비율의 합이 {total:.4f} 다. 1.0 이어야 한다.", subject="split_ratio"
            )
        if any(r <= 0 for r in self.split_ratio):
            raise InvariantViolation(
                "모든 분할은 0보다 커야 한다.", subject="split_ratio"
            )

    @property
    def feature_count(self) -> int:
        return len(self.feature_fields)

    @property
    def gates_passed(self) -> bool:
        """두 게이트를 모두 통과했는가."""
        return self.readiness_certified and self.quality_gate_passed

    @property
    def missing_gates(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not self.readiness_certified:
            missing.append("모듈 1 · 학습 착수 판정")
        if not self.quality_gate_passed:
            missing.append("모듈 2 · Data Quality Gate")
        return tuple(missing)
