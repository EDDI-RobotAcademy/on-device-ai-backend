"""학습 중에 양자화를 가르쳐라. (실습 4-12)

실습 4-7 에서 한 것은 **PTQ**(Post-Training Quantization)다.
다 학습한 모델을 나중에 정수로 바꿨다.

    학습(FP32) → 변환(INT8) → 정확도 확인

이 방법의 좋은 점은 명확하다. **학습 파이프라인을 안 건드린다.**
이미 있는 모델에 바로 적용된다.

문제는 모델이 그 사실을 모른 채 학습했다는 것이다.
FP32 로 배운 미세한 차이가 정수 격자에 눌리면서 사라진다.
비트를 줄일수록 그 손실이 커진다.

**QAT**(Quantization-Aware Training)는 순서를 바꾼다.

    학습(FP32 이지만 순전파에서 INT8 로 눌러 본다) → 변환(INT8)

학습 중에 눌린 값으로 손실을 계산하므로,
모델이 **눌려도 견디는 가중치**를 찾아간다.

대신 대가가 있다.
    - 학습 파이프라인을 하나 더 유지해야 한다
    - 학습이 느려진다
    - 재학습할 때마다 이 절차를 다시 밟아야 한다

그래서 순서가 있다. **PTQ 를 먼저 재 보고, 부족할 때만 QAT 로 간다.**
PTQ 로 충분한 문제에 QAT 를 얹는 것은 비용만 늘리는 일이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


class QuantizationApproach(Enum):
    POST_TRAINING = "POST_TRAINING"
    """다 배운 뒤에 누른다. 학습 파이프라인을 안 건드린다."""

    QUANTIZATION_AWARE = "QUANTIZATION_AWARE"
    """배우는 동안 눌러 본다. 눌려도 견디는 가중치를 찾아간다."""


@dataclass(frozen=True, slots=True)
class QuantizationSpec:
    """몇 비트로, 어떻게 누를 것인가."""

    approach: QuantizationApproach
    bits: int = 8
    per_channel: bool = True
    """채널마다 따로 배율을 잡는가.

    한 배율로 전체를 누르면 큰 채널 하나가 나머지를 다 눌러 버린다.
    """

    def __post_init__(self) -> None:
        if not 2 <= self.bits <= 16:
            raise InvariantViolation(
                "비트 수는 2~16 이어야 한다.", subject="bits"
            )

    @property
    def levels(self) -> int:
        """표현할 수 있는 값의 개수. 8비트면 256, 4비트면 16."""
        return 2**self.bits

    def describe(self) -> str:
        scope = "채널별" if self.per_channel else "전체 공통"
        return f"{self.approach.value} {self.bits}bit ({scope} 배율, {self.levels}단계)"


@dataclass(frozen=True, slots=True)
class QuantizationOutcome:
    """한 가지 방식으로 눌러 본 결과. Infrastructure 가 채운다."""

    spec: QuantizationSpec
    label: str
    baseline_accuracy: float
    quantized_accuracy: float
    baseline_macro_recall: float = 0.0
    quantized_macro_recall: float = 0.0
    training_seconds: float = 0.0
    weight_bytes: int = 0

    def __post_init__(self) -> None:
        for name in (
            "baseline_accuracy",
            "quantized_accuracy",
            "baseline_macro_recall",
            "quantized_macro_recall",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise InvariantViolation(f"{name} 는 0~1 이어야 한다.", subject=name)

    @property
    def accuracy_drop(self) -> float:
        return self.baseline_accuracy - self.quantized_accuracy

    def describe(self) -> str:
        return (
            f"{self.label:<24}{self.spec.bits:>3}bit  "
            f"정확도 {self.quantized_accuracy:.3f} ({-self.accuracy_drop:+.3f})  "
            f"재현율 {self.quantized_macro_recall:.3f}  "
            f"학습 {self.training_seconds:>5.1f}s"
        )


@dataclass(frozen=True, slots=True)
class QuantizationComparison:
    """PTQ 와 QAT 를 나란히 놓은 결과. (실습 4-12)"""

    bits: int
    post_training: QuantizationOutcome
    quantization_aware: QuantizationOutcome

    def __post_init__(self) -> None:
        if (
            self.post_training.spec.approach
            is not QuantizationApproach.POST_TRAINING
        ):
            raise InvariantViolation(
                "첫 칸은 PTQ 여야 한다.", subject="post_training"
            )
        if (
            self.quantization_aware.spec.approach
            is not QuantizationApproach.QUANTIZATION_AWARE
        ):
            raise InvariantViolation(
                "둘째 칸은 QAT 여야 한다.", subject="quantization_aware"
            )

    @property
    def recovered(self) -> float:
        """QAT 가 PTQ 대비 되찾은 정확도."""
        return (
            self.quantization_aware.quantized_accuracy
            - self.post_training.quantized_accuracy
        )

    @property
    def extra_training_seconds(self) -> float:
        return (
            self.quantization_aware.training_seconds
            - self.post_training.training_seconds
        )

    def render(self) -> str:
        header = f"[{self.bits}비트로 누르면]"
        return "\n".join(
            [
                header,
                f"  기준선(FP32)  정확도 "
                f"{self.post_training.baseline_accuracy:.3f}",
                f"  {self.post_training.describe()}",
                f"  {self.quantization_aware.describe()}",
                "",
                f"  QAT 가 되찾은 정확도 {self.recovered:+.3f}  "
                f"(학습 시간 {self.extra_training_seconds:+.1f}s)",
            ]
        )


@dataclass(frozen=True, slots=True)
class QuantizationPolicy:
    """QAT 를 도입할 근거가 있는가. (실습 4-12)"""

    acceptable_drop: float = 0.02
    """PTQ 손실이 이만큼 안쪽이면 **QAT 는 필요 없다.**"""

    min_recovery: float = 0.02
    """QAT 가 이만큼도 못 되찾으면 도입할 이유가 없다."""

    def inspect(self, comparison: QuantizationComparison) -> tuple[Finding, ...]:
        findings: list[Finding] = []
        ptq = comparison.post_training
        qat = comparison.quantization_aware

        if ptq.accuracy_drop <= self.acceptable_drop:
            findings.append(
                Finding(
                    code="QUANT_PTQ_SUFFICIENT",
                    message=(
                        f"PTQ 손실이 {ptq.accuracy_drop:.3f} 다. "
                        "**이 정도면 QAT 를 도입할 이유가 없다** — "
                        "학습 파이프라인을 하나 더 유지하는 비용은 "
                        "재학습할 때마다 계속 나간다."
                    ),
                    severity=Severity.INFO,
                    subject=str(comparison.bits),
                    measured=ptq.accuracy_drop,
                    threshold=self.acceptable_drop,
                )
            )
        elif comparison.recovered < self.min_recovery:
            findings.append(
                Finding(
                    code="QUANT_QAT_DID_NOT_HELP",
                    message=(
                        f"PTQ 가 {ptq.accuracy_drop:.3f} 를 잃었는데 "
                        f"QAT 가 {comparison.recovered:+.3f} 밖에 못 되찾았다. "
                        "**문제가 양자화가 아닐 수 있다** — "
                        "비트 수를 올리거나 구조를 다시 보는 편이 낫다."
                    ),
                    severity=Severity.WARNING,
                    subject=str(comparison.bits),
                    measured=comparison.recovered,
                    threshold=self.min_recovery,
                )
            )
        else:
            findings.append(
                Finding(
                    code="QUANT_QAT_JUSTIFIED",
                    message=(
                        f"PTQ 가 {ptq.accuracy_drop:.3f} 를 잃었고 "
                        f"QAT 가 {comparison.recovered:+.3f} 를 되찾았다. "
                        "**여기서는 QAT 를 도입할 근거가 있다.** "
                        "다만 재학습 절차에 이 단계가 영구히 들어간다."
                    ),
                    severity=Severity.INFO,
                    subject=str(comparison.bits),
                    measured=comparison.recovered,
                    threshold=self.min_recovery,
                )
            )

        if qat.accuracy_drop > self.acceptable_drop:
            findings.append(
                Finding(
                    code="QUANT_STILL_DEGRADED",
                    message=(
                        f"QAT 를 하고도 정확도가 {qat.accuracy_drop:.3f} 낮다. "
                        f"**{comparison.bits}비트가 이 모델에 너무 좁다** — "
                        "비트를 올리거나, 이 정밀도를 포기해야 한다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=str(comparison.bits),
                    measured=qat.accuracy_drop,
                    threshold=self.acceptable_drop,
                )
            )

        return tuple(findings)
