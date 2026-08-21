"""데이터 분할(Value Object + Policy).

실습 1-8 "데이터를 쪼개야 진짜 문제가 보인다".

분할은 나누기가 아니라 **시험 문제를 격리하는 일**이다.
현장 데이터에서 가장 흔한 사고 두 가지가 여기서 발생한다.

    1) 시계열을 무작위로 섞어 나눈다.
       → 10:00:00 이 train, 10:00:01 이 test 로 간다.
       → 사실상 정답을 옆에서 보고 푸는 것이다. 검증 점수는 99%, 현장에서는 실패한다.

    2) 같은 설비/LOT/제품의 사진이 train 과 test 에 흩어진다.
       → 모델은 결함이 아니라 그 설비의 배경을 외운다.

그래서 SplitStrategy 는 취향이 아니라 데이터 구조가 결정한다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from domain.data.inspection import Finding, InspectionKind, InspectionReport, Severity
from domain.data.schema import DataSchema, FieldRole
from domain.shared.errors import InvariantViolation


class SplitStrategy(Enum):
    RANDOM = "RANDOM"
    """행 사이에 순서도 그룹도 없을 때만 안전하다."""

    TIME_ORDERED = "TIME_ORDERED"
    """과거로 배우고 미래로 평가한다. 시계열의 기본값."""

    GROUP_HOLDOUT = "GROUP_HOLDOUT"
    """같은 그룹(설비/LOT/제품)은 통째로 한쪽에만 넣는다."""


@dataclass(frozen=True, slots=True)
class SplitRatio:
    train: float
    validation: float
    test: float

    def __post_init__(self) -> None:
        for name in ("train", "validation", "test"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise InvariantViolation(
                    f"{name} 비율은 0 과 1 사이여야 한다.", subject=name
                )
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-9:
            raise InvariantViolation(
                f"분할 비율의 합이 {total:.4f} 다. 1.0 이어야 한다.", subject="SplitRatio"
            )

    @classmethod
    def of(cls, train: float, validation: float, test: float) -> SplitRatio:
        return cls(train=train, validation=validation, test=test)


@dataclass(frozen=True, slots=True)
class PartitionPlan:
    """어떻게 나눌 것인가에 대한 계획."""

    strategy: SplitStrategy
    ratio: SplitRatio
    time_field: str | None = None
    group_field: str | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.strategy is SplitStrategy.TIME_ORDERED and not self.time_field:
            raise InvariantViolation(
                "시간 기준 분할인데 시간 필드가 없다.", subject="time_field"
            )
        if self.strategy is SplitStrategy.GROUP_HOLDOUT and not self.group_field:
            raise InvariantViolation(
                "그룹 분할인데 그룹 필드가 없다.", subject="group_field"
            )

    def validate_against(self, schema: DataSchema) -> None:
        """계획이 이 스키마에서 성립하는지 확인한다."""
        if self.time_field is not None:
            spec = schema.field_of(self.time_field)
            if spec.role is not FieldRole.TIME_INDEX:
                raise InvariantViolation(
                    f"'{self.time_field}' 은 TIME_INDEX 가 아니다.", subject=self.time_field
                )
        if self.group_field is not None:
            spec = schema.field_of(self.group_field)
            if spec.role is not FieldRole.GROUP:
                raise InvariantViolation(
                    f"'{self.group_field}' 은 GROUP 이 아니다.", subject=self.group_field
                )
        if self.strategy is SplitStrategy.RANDOM and schema.time_index is not None:
            raise InvariantViolation(
                "시간축이 있는 데이터를 무작위로 나누면 미래가 학습에 섞인다. "
                "TIME_ORDERED 를 쓴다.",
                subject="strategy",
            )


@dataclass(frozen=True, slots=True)
class PartitionMeasurement:
    """실제로 나눈 결과에 대해 Infrastructure 가 측정한 값."""

    train_count: int
    validation_count: int
    test_count: int
    overlapping_group_count: int = 0
    """train 과 test 양쪽에 동시에 등장한 그룹 수."""

    time_overlap_seconds: float = 0.0
    """train 의 마지막 시각이 test 의 첫 시각을 넘어선 정도."""

    class_distribution: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    """split 이름 → (클래스 → 개수)."""

    def __post_init__(self) -> None:
        for name in ("train_count", "validation_count", "test_count"):
            if getattr(self, name) < 0:
                raise InvariantViolation(f"{name} 는 음수일 수 없다.", subject=name)

    @property
    def total_count(self) -> int:
        return self.train_count + self.validation_count + self.test_count

    def ratio_of(self, split: str) -> float:
        if self.total_count == 0:
            return 0.0
        return getattr(self, f"{split}_count") / self.total_count

    def class_ratio(self, split: str) -> dict[str, float]:
        counts = self.class_distribution.get(split, {})
        total = sum(counts.values())
        if total == 0:
            return {}
        return {name: c / total for name, c in counts.items()}


@dataclass(frozen=True, slots=True)
class PartitionPolicy:
    """분할 결과를 신뢰해도 되는지에 대한 기준."""

    min_samples_per_split: int = 1
    ratio_tolerance: float = 0.05
    max_class_ratio_gap: float = 0.1
    """train 과 test 의 클래스 구성비 차이 허용치."""

    def inspect(
        self, plan: PartitionPlan, measurement: PartitionMeasurement
    ) -> InspectionReport:
        findings: list[Finding] = []

        for split in ("train", "validation", "test"):
            count = getattr(measurement, f"{split}_count")
            if count < self.min_samples_per_split:
                findings.append(
                    Finding(
                        code="PARTITION_EMPTY_SPLIT",
                        message=f"{split} 분할에 표본이 부족하다.",
                        severity=Severity.CRITICAL,
                        subject=split,
                        measured=float(count),
                        threshold=float(self.min_samples_per_split),
                    )
                )
                continue
            expected = getattr(plan.ratio, split)
            actual = measurement.ratio_of(split)
            if abs(actual - expected) > self.ratio_tolerance:
                findings.append(
                    Finding(
                        code="PARTITION_RATIO_DRIFT",
                        message=f"{split} 비율이 계획과 다르다.",
                        severity=Severity.WARNING,
                        subject=split,
                        measured=actual,
                        threshold=expected,
                    )
                )

        if measurement.overlapping_group_count > 0:
            findings.append(
                Finding(
                    code="PARTITION_GROUP_LEAKAGE",
                    message=(
                        f"{measurement.overlapping_group_count} 개 그룹이 train 과 test 에 "
                        "동시에 있다. 모델은 결함이 아니라 그 그룹을 외운다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=plan.group_field or "group",
                    measured=float(measurement.overlapping_group_count),
                    threshold=0.0,
                )
            )

        if measurement.time_overlap_seconds > 0:
            findings.append(
                Finding(
                    code="PARTITION_TIME_LEAKAGE",
                    message=(
                        "train 구간이 test 구간을 침범했다. 미래를 보고 과거를 맞히는 셈이다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=plan.time_field or "time",
                    measured=measurement.time_overlap_seconds,
                    threshold=0.0,
                )
            )

        train_ratio = measurement.class_ratio("train")
        test_ratio = measurement.class_ratio("test")
        for name in sorted(set(train_ratio) | set(test_ratio)):
            gap = abs(train_ratio.get(name, 0.0) - test_ratio.get(name, 0.0))
            if gap > self.max_class_ratio_gap:
                findings.append(
                    Finding(
                        code="PARTITION_CLASS_SKEW",
                        message=(
                            f"'{name}' 클래스 비중이 train 과 test 에서 크게 다르다. "
                            "평가 점수를 그대로 믿을 수 없다."
                        ),
                        severity=Severity.WARNING,
                        subject=name,
                        measured=gap,
                        threshold=self.max_class_ratio_gap,
                    )
                )

        return InspectionReport(kind=InspectionKind.PARTITION, findings=tuple(findings))
