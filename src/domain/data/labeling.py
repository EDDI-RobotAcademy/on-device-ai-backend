"""정상과 이상의 정의(Value Object + Policy).

실습 1-6 "정상과 이상을 누가 결정하는가?".

라벨은 데이터에 들어 있지 않다. 사람이 넣는 것이다.
따라서 라벨 품질 문제는 대부분 *정의가 없거나, 정의가 사람마다 다른* 데서 나온다.

    "이 정도 표면 흠집이 불량인가?"
    → 품질팀은 불량, 생산팀은 양품. 두 사람이 같은 사진에 다른 라벨을 붙인다.
    → 모델은 그 모순을 그대로 학습하고, 현장에서는 "AI가 오락가락한다"고 말한다.

그래서 LabelDefinition 은 이름만으로 만들 수 없다. 판단 기준(meaning)과
누가 결정했는지(decided_by)를 반드시 요구한다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from domain.data.errors import UnknownField
from domain.data.inspection import Finding, InspectionKind, InspectionReport, Severity
from domain.shared.errors import InvariantViolation


@dataclass(frozen=True, slots=True)
class LabelDefinition:
    """클래스 하나의 정의."""

    name: str
    meaning: str
    """다른 사람이 읽고 같은 판단을 내릴 수 있는 문장."""

    decided_by: str
    """이 기준을 확정한 주체. 예: "품질보증팀", "설비 알람 규칙 R-12"."""

    examples: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise InvariantViolation("라벨 이름은 비어 있을 수 없다.", subject="name")
        if len(self.meaning.strip()) < 5:
            raise InvariantViolation(
                f"'{self.name}' 의 판단 기준이 없다. 기준 없는 라벨은 사람마다 달라진다.",
                subject=self.name,
            )
        if not self.decided_by.strip():
            raise InvariantViolation(
                f"'{self.name}' 을 누가 정했는지 없다. 분쟁이 나면 되돌릴 근거가 없다.",
                subject=self.name,
            )


@dataclass(frozen=True, slots=True)
class LabelSpace:
    """이 Dataset 이 인정하는 클래스의 전체 집합."""

    field_name: str
    definitions: tuple[LabelDefinition, ...]

    def __post_init__(self) -> None:
        if not self.field_name.strip():
            raise InvariantViolation("라벨 필드 이름이 필요하다.", subject="field_name")
        if len(self.definitions) < 2:
            raise InvariantViolation(
                "클래스가 하나뿐이면 분류 문제가 성립하지 않는다.", subject="definitions"
            )
        names = [d.name for d in self.definitions]
        if len(names) != len(set(names)):
            raise InvariantViolation("중복된 클래스 이름이 있다.", subject="definitions")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(d.name for d in self.definitions)

    def definition_of(self, name: str) -> LabelDefinition:
        for d in self.definitions:
            if d.name == name:
                return d
        raise UnknownField(f"'{name}' 은 정의되지 않은 클래스다.", subject=name)

    def unknown_labels(self, observed: tuple[str, ...]) -> tuple[str, ...]:
        """데이터에는 있는데 정의에는 없는 라벨."""
        known = set(self.names)
        return tuple(sorted({o for o in observed if o not in known}))


@dataclass(frozen=True, slots=True)
class LabelAgreementMeasurement:
    """라벨링 결과에 대해 Infrastructure 가 측정한 값."""

    class_counts: Mapping[str, int]
    annotator_count: int = 1
    reviewed_sample_count: int = 0
    """두 명 이상이 겹쳐서 라벨링한 표본 수."""

    disagreement_count: int = 0
    """그중 판단이 갈린 표본 수."""

    unlabeled_count: int = 0

    def __post_init__(self) -> None:
        if self.annotator_count < 1:
            raise InvariantViolation("작업자 수는 1 이상이어야 한다.", subject="annotator_count")
        if any(v < 0 for v in self.class_counts.values()):
            raise InvariantViolation("클래스 개수는 음수일 수 없다.", subject="class_counts")
        if self.disagreement_count > self.reviewed_sample_count:
            raise InvariantViolation(
                "불일치 수가 교차 검토 표본 수보다 클 수 없다.", subject="disagreement_count"
            )

    @property
    def observed_labels(self) -> tuple[str, ...]:
        return tuple(self.class_counts.keys())

    @property
    def labeled_count(self) -> int:
        return sum(self.class_counts.values())

    @property
    def total_count(self) -> int:
        return self.labeled_count + self.unlabeled_count

    @property
    def unlabeled_ratio(self) -> float:
        return self.unlabeled_count / self.total_count if self.total_count else 0.0

    @property
    def agreement_ratio(self) -> float:
        """교차 검토된 표본 중 판단이 일치한 비율. 검토가 없으면 1.0 이 아니라 0.0 이다."""
        if self.reviewed_sample_count == 0:
            return 0.0
        return 1.0 - self.disagreement_count / self.reviewed_sample_count

    @property
    def imbalance_ratio(self) -> float:
        """가장 많은 클래스 / 가장 적은 클래스."""
        counts = [c for c in self.class_counts.values()]
        if not counts:
            return 0.0
        smallest = min(counts)
        if smallest == 0:
            return float("inf")
        return max(counts) / smallest


@dataclass(frozen=True, slots=True)
class LabelPolicy:
    """라벨을 믿어도 되는지에 대한 기준."""

    min_agreement_ratio: float = 0.9
    min_samples_per_class: int = 30
    max_unlabeled_ratio: float = 0.0
    max_imbalance_ratio: float = 10.0
    require_cross_review: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_agreement_ratio <= 1.0:
            raise InvariantViolation(
                "min_agreement_ratio 는 0~1 이어야 한다.", subject="min_agreement_ratio"
            )
        if self.min_samples_per_class < 1:
            raise InvariantViolation(
                "클래스당 최소 표본 수는 1 이상이어야 한다.", subject="min_samples_per_class"
            )

    def inspect(
        self, space: LabelSpace, measurement: LabelAgreementMeasurement
    ) -> InspectionReport:
        findings: list[Finding] = []

        unknown = space.unknown_labels(measurement.observed_labels)
        for name in unknown:
            findings.append(
                Finding(
                    code="LABEL_UNDEFINED",
                    message=f"데이터에 '{name}' 라벨이 있으나 정의된 클래스가 아니다.",
                    severity=Severity.CRITICAL,
                    subject=name,
                    measured=float(measurement.class_counts.get(name, 0)),
                )
            )

        for name in space.names:
            count = measurement.class_counts.get(name, 0)
            if count == 0:
                findings.append(
                    Finding(
                        code="LABEL_CLASS_EMPTY",
                        message="정의만 있고 실제 표본이 하나도 없는 클래스다.",
                        severity=Severity.CRITICAL,
                        subject=name,
                        measured=0.0,
                        threshold=float(self.min_samples_per_class),
                    )
                )
            elif count < self.min_samples_per_class:
                findings.append(
                    Finding(
                        code="LABEL_CLASS_TOO_FEW",
                        message="표본이 너무 적어 이 클래스는 사실상 학습되지 않는다.",
                        severity=Severity.WARNING,
                        subject=name,
                        measured=float(count),
                        threshold=float(self.min_samples_per_class),
                    )
                )

        if measurement.unlabeled_ratio > self.max_unlabeled_ratio:
            findings.append(
                Finding(
                    code="LABEL_MISSING",
                    message="라벨이 비어 있는 표본이 있다.",
                    severity=Severity.CRITICAL,
                    subject=space.field_name,
                    measured=measurement.unlabeled_ratio,
                    threshold=self.max_unlabeled_ratio,
                )
            )

        if self.require_cross_review:
            if measurement.reviewed_sample_count == 0:
                findings.append(
                    Finding(
                        code="LABEL_NO_CROSS_REVIEW",
                        message=(
                            "교차 검토가 한 건도 없다. 라벨이 일관적인지 확인할 방법이 없다."
                        ),
                        severity=Severity.CRITICAL,
                        subject=space.field_name,
                        measured=0.0,
                        threshold=1.0,
                    )
                )
            elif measurement.agreement_ratio < self.min_agreement_ratio:
                findings.append(
                    Finding(
                        code="LABEL_DISAGREEMENT",
                        message=(
                            f"작업자 {measurement.annotator_count} 명의 판단이 갈린다. "
                            "모델이 배우는 것은 결함이 아니라 사람의 기준 차이다."
                        ),
                        severity=Severity.CRITICAL,
                        subject=space.field_name,
                        measured=measurement.agreement_ratio,
                        threshold=self.min_agreement_ratio,
                    )
                )

        if measurement.imbalance_ratio > self.max_imbalance_ratio:
            findings.append(
                Finding(
                    code="LABEL_IMBALANCED",
                    message=(
                        "클래스 불균형이 크다. 다수 클래스만 찍어도 정확도가 높게 나온다."
                    ),
                    severity=Severity.WARNING,
                    subject=space.field_name,
                    measured=measurement.imbalance_ratio,
                    threshold=self.max_imbalance_ratio,
                )
            )

        return InspectionReport(kind=InspectionKind.LABEL_SPACE, findings=tuple(findings))
