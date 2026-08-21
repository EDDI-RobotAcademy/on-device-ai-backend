"""학습 데이터와 현실의 거리(Value Object + Policy).

실습 1-9 "Training Data와 Reality는 왜 다른가?".

학습 데이터는 언제나 과거이고, 대체로 *잘 정리된 과거*다.
    - 여름에 모은 데이터로 학습하고 겨울 현장에 배포한다.
    - 신형 설비 3호기 데이터만 모으고 노후한 1호기에 배포한다.
    - 정상 위주로 모으고, 정작 잡아야 할 이상은 거의 없다.

여기서 재는 것은 "데이터가 깨졌는가"가 아니라 "데이터가 현실을 대표하는가"다.
같은 도구(PSI)를 운영 단계에서 다시 쓰면 그게 Data Drift 감시가 된다.
(운영 모듈의 DriftReport 와 이어지는 지점이다.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from domain.data.inspection import Finding, InspectionKind, InspectionReport, Severity
from domain.shared.errors import InvariantViolation


@dataclass(frozen=True, slots=True)
class FieldDistributionShift:
    """필드 하나에 대한 분포 비교 결과."""

    field_name: str
    psi: float
    """Population Stability Index. 업계 관행: 0.1 미만 안정 / 0.25 이상 심각."""

    reference_mean: float | None = None
    observed_mean: float | None = None
    coverage_ratio: float = 1.0
    """현실에서 관측된 값 범위 중 학습 데이터가 덮고 있는 비율."""

    def __post_init__(self) -> None:
        if self.psi < 0:
            raise InvariantViolation("PSI 는 음수일 수 없다.", subject=self.field_name)
        if not 0.0 <= self.coverage_ratio <= 1.0:
            raise InvariantViolation(
                "coverage_ratio 는 0~1 이어야 한다.", subject=self.field_name
            )

    @property
    def mean_shift(self) -> float | None:
        if self.reference_mean is None or self.observed_mean is None:
            return None
        return self.observed_mean - self.reference_mean


@dataclass(frozen=True, slots=True)
class RepresentativenessMeasurement:
    """학습 데이터(reference) 와 현실 표본(observed) 을 비교한 측정값."""

    reference_label: str
    observed_label: str
    field_shifts: tuple[FieldDistributionShift, ...] = field(default_factory=tuple)
    unseen_category_ratio: float = 0.0
    """현실에는 있는데 학습 데이터에는 한 번도 없던 범주(설비/제품/조건)의 비율."""

    observed_sample_count: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.unseen_category_ratio <= 1.0:
            raise InvariantViolation(
                "unseen_category_ratio 는 0~1 이어야 한다.", subject="unseen_category_ratio"
            )

    @property
    def worst_psi(self) -> float:
        return max((s.psi for s in self.field_shifts), default=0.0)

    @property
    def worst_field(self) -> str | None:
        if not self.field_shifts:
            return None
        return max(self.field_shifts, key=lambda s: s.psi).field_name


@dataclass(frozen=True, slots=True)
class RepresentativenessPolicy:
    """학습 데이터가 현실을 대표한다고 말할 수 있는 기준."""

    psi_warning_threshold: float = 0.10
    psi_critical_threshold: float = 0.25
    min_coverage_ratio: float = 0.9
    max_unseen_category_ratio: float = 0.05
    min_observed_sample_count: int = 100

    def __post_init__(self) -> None:
        if self.psi_warning_threshold >= self.psi_critical_threshold:
            raise InvariantViolation(
                "경고 임계가 심각 임계보다 작아야 한다.", subject="psi thresholds"
            )

    def inspect(self, measurement: RepresentativenessMeasurement) -> InspectionReport:
        m = measurement
        findings: list[Finding] = []

        if m.observed_sample_count < self.min_observed_sample_count:
            findings.append(
                Finding(
                    code="REPR_SAMPLE_TOO_SMALL",
                    message=(
                        "현실 표본이 너무 적어 비교 자체를 신뢰할 수 없다. "
                        "이 상태의 '이상 없음'은 근거가 아니다."
                    ),
                    severity=Severity.WARNING,
                    subject=m.observed_label,
                    measured=float(m.observed_sample_count),
                    threshold=float(self.min_observed_sample_count),
                )
            )

        for shift in m.field_shifts:
            if shift.psi >= self.psi_critical_threshold:
                findings.append(
                    Finding(
                        code="REPR_DISTRIBUTION_SHIFTED",
                        message=(
                            f"'{shift.field_name}' 의 분포가 학습 시점과 크게 다르다. "
                            "이 데이터로 학습한 모델은 현재 현장을 본 적이 없다."
                        ),
                        severity=Severity.CRITICAL,
                        subject=shift.field_name,
                        measured=shift.psi,
                        threshold=self.psi_critical_threshold,
                    )
                )
            elif shift.psi >= self.psi_warning_threshold:
                findings.append(
                    Finding(
                        code="REPR_DISTRIBUTION_DRIFTING",
                        message=f"'{shift.field_name}' 의 분포가 이동하고 있다.",
                        severity=Severity.WARNING,
                        subject=shift.field_name,
                        measured=shift.psi,
                        threshold=self.psi_warning_threshold,
                    )
                )

            if shift.coverage_ratio < self.min_coverage_ratio:
                findings.append(
                    Finding(
                        code="REPR_COVERAGE_GAP",
                        message=(
                            f"'{shift.field_name}' 에서 현실이 학습 데이터 범위 밖으로 나간다. "
                            "모델은 그 구간을 외삽(extrapolation)으로 찍게 된다."
                        ),
                        severity=Severity.CRITICAL,
                        subject=shift.field_name,
                        measured=shift.coverage_ratio,
                        threshold=self.min_coverage_ratio,
                    )
                )

        if m.unseen_category_ratio > self.max_unseen_category_ratio:
            findings.append(
                Finding(
                    code="REPR_UNSEEN_CATEGORY",
                    message=(
                        "학습 데이터에 한 번도 없던 조건이 현실에 존재한다. "
                        "예: 학습에 없던 설비 호기, 신규 제품 코드."
                    ),
                    severity=Severity.CRITICAL,
                    subject="category",
                    measured=m.unseen_category_ratio,
                    threshold=self.max_unseen_category_ratio,
                )
            )

        return InspectionReport(
            kind=InspectionKind.REPRESENTATIVENESS, findings=tuple(findings)
        )
