"""이상치 — 값이 말이 안 된다. (실습 2-3)

이상치는 "제거 대상"이 아니다. **이상치가 바로 우리가 찾는 사고일 수 있다.**

    NORMAL 라벨 구간에 있는 이상치  → 라벨이 틀렸거나, 아직 아무도 모르는 사고다
    FAULT 라벨 구간에 있는 이상치   → 그게 신호다. 지우면 배울 것이 없어진다

그래서 이 검사는 라벨과 함께 본다. 라벨 없이 이상치를 지우는 것이
데이터 품질 작업에서 가장 흔하고 가장 비싼 실수다.

또 하나: 평균과 표준편차로 이상치를 찾으면, 이상치가 많을수록
평균과 표준편차 자체가 오염되어 이상치를 못 찾게 된다.
그래서 MAD(중앙값 절대편차) 같은 강건한 척도를 함께 본다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from domain.data_quality.dimensions import (
    DimensionResult,
    QualityDimension,
    QualityScore,
    deduct,
)
from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


@dataclass(frozen=True, slots=True)
class FieldOutliers:
    """열 하나의 이상치 실태."""

    field_name: str
    total_count: int
    z_outlier_count: int = 0
    """|z| > 3. 평균·표준편차 기반 — 이상치가 많으면 스스로 오염된다."""

    mad_outlier_count: int = 0
    """수정 z-score > 3.5. 중앙값 기반이라 강건하다."""

    out_of_physical_range_count: int = 0
    rate_violation_count: int = 0
    """직전 표본 대비 변화율이 물리적으로 불가능한 표본 수."""

    max_abs_z: float = 0.0
    outliers_by_label: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total_count < 0:
            raise InvariantViolation("total_count 는 음수일 수 없다.", subject=self.field_name)
        for name in (
            "z_outlier_count",
            "mad_outlier_count",
            "out_of_physical_range_count",
            "rate_violation_count",
        ):
            value = getattr(self, name)
            if value < 0 or value > self.total_count:
                raise InvariantViolation(
                    f"{name}({value}) 이 total_count({self.total_count}) 범위를 벗어났다.",
                    subject=self.field_name,
                )

    def _ratio(self, count: int) -> float:
        return count / self.total_count if self.total_count else 0.0

    @property
    def z_outlier_ratio(self) -> float:
        return self._ratio(self.z_outlier_count)

    @property
    def mad_outlier_ratio(self) -> float:
        return self._ratio(self.mad_outlier_count)

    @property
    def out_of_range_ratio(self) -> float:
        return self._ratio(self.out_of_physical_range_count)

    @property
    def rate_violation_ratio(self) -> float:
        return self._ratio(self.rate_violation_count)

    @property
    def masking_gap(self) -> int:
        """MAD 는 잡았는데 z-score 는 놓친 표본 수.

        이 값이 크면 평균·표준편차가 이미 이상치에 오염되었다는 뜻이다.
        """
        return max(self.mad_outlier_count - self.z_outlier_count, 0)

    def outlier_share_of(self, label: str) -> float:
        total = sum(self.outliers_by_label.values())
        if total == 0:
            return 0.0
        return self.outliers_by_label.get(label, 0) / total


@dataclass(frozen=True, slots=True)
class OutlierMeasurement:
    fields: tuple[FieldOutliers, ...] = field(default_factory=tuple)

    def field_of(self, name: str) -> FieldOutliers | None:
        for item in self.fields:
            if item.field_name == name:
                return item
        return None

    @property
    def worst_field(self) -> str | None:
        if not self.fields:
            return None
        return max(self.fields, key=lambda f: f.mad_outlier_ratio).field_name


@dataclass(frozen=True, slots=True)
class ValidityPolicy:
    max_outlier_ratio: float = 0.01
    max_out_of_range_ratio: float = 0.0
    max_rate_violation_ratio: float = 0.005
    """0 이 아니다. 설비가 실제로 정지하면 값은 실제로 급변한다.
    정상 데이터에도 급변은 있고, 기준은 그 사실을 인정한 자리에 있어야 한다."""
    max_masking_gap_ratio: float = 0.005
    """z-score 가 놓친 비율이 이만큼을 넘으면 평균·표준편차를 믿을 수 없다."""

    max_normal_label_outlier_share: float = 0.7
    """이상치의 70% 이상이 '정상' 라벨에 있으면 라벨을 의심한다."""

    normal_label: str | None = "NORMAL"
    """무엇을 '아무 일도 없었음'으로 볼 것인가.

    이것은 측정 설정이 아니라 **도메인 지식**이다. 라인마다 라벨 이름이 다르다.
    그래서 측정기가 아니라 Policy 가 들고 있다."""

    def __post_init__(self) -> None:
        for name in (
            "max_outlier_ratio",
            "max_out_of_range_ratio",
            "max_rate_violation_ratio",
            "max_masking_gap_ratio",
            "max_normal_label_outlier_share",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise InvariantViolation(f"{name} 는 0~1 비율이어야 한다.", subject=name)

    def evaluate(self, measurement: OutlierMeasurement) -> DimensionResult:
        """점수는 가장 나쁜 채널로 낸다. 한 채널이 못 쓰게 되면 그 채널은 못 쓴다."""
        findings: list[Finding] = []
        per_field: dict[str, float] = {}

        def penalize(name: str, amount: float) -> None:
            per_field[name] = per_field.get(name, 0.0) + amount

        for item in measurement.fields:
            per_field.setdefault(item.field_name, 0.0)
            if item.out_of_range_ratio > self.max_out_of_range_ratio:
                findings.append(
                    Finding(
                        code="VALIDITY_OUT_OF_RANGE",
                        message="물리적으로 불가능한 값이 남아 있다.",
                        severity=Severity.CRITICAL,
                        subject=item.field_name,
                        measured=item.out_of_range_ratio,
                        threshold=self.max_out_of_range_ratio,
                    )
                )
                penalize(
                    item.field_name,
                    deduct(
                        item.out_of_range_ratio,
                        tolerance=self.max_out_of_range_ratio,
                        cap=0.05,
                        weight=30.0,
                    ),
                )

            if item.mad_outlier_ratio > self.max_outlier_ratio:
                findings.append(
                    Finding(
                        code="VALIDITY_OUTLIER_RATIO",
                        message=(
                            f"강건 척도(MAD) 기준 이상치가 {item.mad_outlier_count} 건이다. "
                            "지우기 전에 이것이 사고인지 오류인지부터 확인해야 한다."
                        ),
                        severity=Severity.WARNING,
                        subject=item.field_name,
                        measured=item.mad_outlier_ratio,
                        threshold=self.max_outlier_ratio,
                    )
                )
                penalize(
                    item.field_name,
                    deduct(
                        item.mad_outlier_ratio,
                        tolerance=self.max_outlier_ratio,
                        cap=0.10,
                        weight=30.0,
                    ),
                )

            masking_ratio = (
                item.masking_gap / item.total_count if item.total_count else 0.0
            )
            if masking_ratio > self.max_masking_gap_ratio:
                findings.append(
                    Finding(
                        code="VALIDITY_ZSCORE_MASKED",
                        message=(
                            f"MAD 는 {item.mad_outlier_count} 건을 잡았는데 z-score 는 "
                            f"{item.z_outlier_count} 건만 잡았다. "
                            "평균·표준편차가 이미 이상치에 오염되었다."
                        ),
                        severity=Severity.WARNING,
                        subject=item.field_name,
                        measured=masking_ratio,
                        threshold=self.max_masking_gap_ratio,
                    )
                )
                penalize(
                    item.field_name,
                    deduct(
                        masking_ratio,
                        tolerance=self.max_masking_gap_ratio,
                        cap=0.05,
                        weight=10.0,
                    ),
                )

            if item.rate_violation_ratio > self.max_rate_violation_ratio:
                findings.append(
                    Finding(
                        code="VALIDITY_RATE_VIOLATION",
                        message=(
                            "직전 표본 대비 변화가 물리적으로 불가능하다. "
                            "값 하나만 보면 정상 범위 안이라 단변량 검사로는 잡히지 않는다."
                        ),
                        severity=Severity.CRITICAL,
                        subject=item.field_name,
                        measured=item.rate_violation_ratio,
                        threshold=self.max_rate_violation_ratio,
                    )
                )
                penalize(
                    item.field_name,
                    deduct(
                        item.rate_violation_ratio,
                        tolerance=self.max_rate_violation_ratio,
                        cap=0.02,
                        weight=20.0,
                    ),
                )

            normal = self.normal_label
            if normal and item.mad_outlier_count > 0:
                share = item.outlier_share_of(normal)
                if share > self.max_normal_label_outlier_share:
                    findings.append(
                        Finding(
                            code="VALIDITY_OUTLIER_IN_NORMAL",
                            message=(
                                f"이상치의 {share:.0%} 가 '{normal}' 라벨 구간에 있다. "
                                "라벨이 틀렸거나, 아직 아무도 사고로 기록하지 않은 사고다. "
                                "어느 쪽이든 지우고 넘어갈 일이 아니다."
                            ),
                            severity=Severity.CRITICAL,
                            subject=item.field_name,
                            measured=share,
                            threshold=self.max_normal_label_outlier_share,
                        )
                    )
                penalize(
                    item.field_name,
                    deduct(
                        share,
                        tolerance=self.max_normal_label_outlier_share,
                        cap=1.0,
                        weight=10.0,
                    ),
                )

        return DimensionResult(
            dimension=QualityDimension.VALIDITY,
            score=QualityScore.from_deductions([max(per_field.values(), default=0.0)]),
            findings=tuple(findings),
        )
