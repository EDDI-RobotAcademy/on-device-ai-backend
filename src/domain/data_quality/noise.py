"""잡음 — 신호가 잡음에 묻혔다. (실습 2-6)

모델은 우리가 가르친 것을 배운다. 잡음을 남겨 두면 잡음을 배운다.

그런데 반대쪽 함정이 더 위험하다. **과하게 매끈하게 만들면 이상 징후 자체가 지워진다.**
전력 데이터에서 우리가 찾는 것은 대부분 짧고 뾰족한 사건이다.
이동평균 창을 키울수록 그 사건은 사라진다.

그래서 이 검사는 두 방향을 함께 본다.
    SNR 이 너무 낮다        → 잡음이 신호를 덮었다
    고주파 성분이 너무 없다  → 이미 누군가 과하게 매끈하게 만들었다
"""

from __future__ import annotations

import math
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
class FieldNoise:
    """열 하나의 잡음 실태."""

    field_name: str
    signal_power: float
    """추세 성분의 분산."""

    noise_power: float
    """추세를 뺀 잔차의 분산."""

    high_frequency_ratio: float = 0.0
    """전체 변동 중 표본 간 급변이 차지하는 비율."""

    reversal_ratio: float = 0.0
    """연속 차분의 부호가 뒤집히는 비율. 1에 가까울수록 톱니 모양이다."""

    def __post_init__(self) -> None:
        if self.signal_power < 0 or self.noise_power < 0:
            raise InvariantViolation("전력(분산)은 음수일 수 없다.", subject=self.field_name)
        for name in ("high_frequency_ratio", "reversal_ratio"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise InvariantViolation(
                    f"{name} 는 0~1 이어야 한다.", subject=self.field_name
                )

    @property
    def snr_db(self) -> float:
        """신호 대 잡음비 (dB). 클수록 깨끗하다."""
        if self.noise_power <= 0:
            return float("inf")
        if self.signal_power <= 0:
            return float("-inf")
        return 10.0 * math.log10(self.signal_power / self.noise_power)

    @property
    def is_oversmoothed(self) -> bool:
        """변동이 거의 없다 — 원본이 아니라 가공된 값일 수 있다."""
        return self.high_frequency_ratio < 1e-6


@dataclass(frozen=True, slots=True)
class NoiseMeasurement:
    fields: tuple[FieldNoise, ...] = field(default_factory=tuple)

    def field_of(self, name: str) -> FieldNoise | None:
        for item in self.fields:
            if item.field_name == name:
                return item
        return None

    @property
    def worst_snr_db(self) -> float:
        return min((f.snr_db for f in self.fields), default=float("inf"))

    @property
    def worst_field(self) -> str | None:
        if not self.fields:
            return None
        return min(self.fields, key=lambda f: f.snr_db).field_name


@dataclass(frozen=True, slots=True)
class NoisePolicy:
    min_snr_db: float = 20.0
    max_reversal_ratio: float = 0.75
    """백색잡음의 기대값이 약 0.67 이다. 0.75 를 넘으면 거의 매 표본 방향이 바뀐다는 뜻으로,
    전원 주파수와 샘플링 주파수의 간섭(aliasing) 같은 계통 문제를 의심해야 한다."""

    min_high_frequency_ratio: float = 1e-5
    """이보다 낮으면 이미 과하게 평활화된 데이터다."""

    def evaluate(self, measurement: NoiseMeasurement) -> DimensionResult:
        """점수는 가장 잡음이 심한 채널로 낸다."""
        findings: list[Finding] = []
        per_field: dict[str, float] = {}

        def penalize(name: str, amount: float) -> None:
            per_field[name] = per_field.get(name, 0.0) + amount

        for item in measurement.fields:
            per_field.setdefault(item.field_name, 0.0)
            if item.snr_db < self.min_snr_db:
                findings.append(
                    Finding(
                        code="NOISE_SNR_LOW",
                        message=(
                            f"신호 대 잡음비가 {item.snr_db:.1f}dB 다. "
                            "이 상태로 학습하면 모델이 잡음의 모양을 외운다."
                        ),
                        severity=Severity.CRITICAL
                        if item.snr_db < self.min_snr_db - 10
                        else Severity.WARNING,
                        subject=item.field_name,
                        measured=item.snr_db,
                        threshold=self.min_snr_db,
                    )
                )
                penalize(
                    item.field_name,
                    deduct(
                        self.min_snr_db - item.snr_db,
                        tolerance=0.0,
                        cap=30.0,
                        weight=55.0,
                    ),
                )

            if item.reversal_ratio > self.max_reversal_ratio:
                findings.append(
                    Finding(
                        code="NOISE_JITTER",
                        message=(
                            "값이 매 표본마다 방향을 바꾼다(톱니). "
                            "센서 진동이거나 통신 잡음이다."
                        ),
                        severity=Severity.WARNING,
                        subject=item.field_name,
                        measured=item.reversal_ratio,
                        threshold=self.max_reversal_ratio,
                    )
                )
                penalize(
                    item.field_name,
                    deduct(
                        item.reversal_ratio,
                        tolerance=self.max_reversal_ratio,
                        cap=1.0,
                        weight=25.0,
                    ),
                )

            if item.high_frequency_ratio < self.min_high_frequency_ratio:
                findings.append(
                    Finding(
                        code="NOISE_OVERSMOOTHED",
                        message=(
                            "고주파 성분이 거의 없다. 누군가 이미 평활화한 데이터일 수 있다. "
                            "짧고 뾰족한 이상 징후는 그 과정에서 사라진다."
                        ),
                        severity=Severity.CRITICAL,
                        subject=item.field_name,
                        measured=item.high_frequency_ratio,
                        threshold=self.min_high_frequency_ratio,
                    )
                )
                penalize(item.field_name, 20.0)

        return DimensionResult(
            dimension=QualityDimension.NOISE,
            score=QualityScore.from_deductions([max(per_field.values(), default=0.0)]),
            findings=tuple(findings),
        )
