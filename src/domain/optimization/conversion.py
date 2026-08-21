"""변환은 '성공했다'로 끝나지 않는다. (실습 4-2, 4-3, 4-4)

변환 도구는 대개 성공한다. 축 순서가 틀려도, 연산자가 근사로 바뀌어도 성공한다.
**그래서 변환 뒤에 반드시 확인해야 한다 — 같은 입력에 같은 답을 내는가.**

확인하는 방법은 정밀도에 따라 다르다.

    FP32 변환   숫자가 거의 그대로여야 한다        → max|diff| 로 본다
    FP16 변환   소수점 아래가 조금 흔들린다         → 느슨한 max|diff|
    INT8 양자화 숫자는 크게 달라진다                → **argmax 가 같은지**로 본다
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.optimization.runtime import Precision, RuntimeTarget
from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


@dataclass(frozen=True, slots=True)
class NumericalEquivalence:
    """변환 전후의 출력을 맞대어 본 결과."""

    sample_count: int
    max_abs_diff: float
    mean_abs_diff: float
    agreement_ratio: float
    """argmax 가 일치한 표본의 비율. 분류 모델에서 실제로 중요한 것은 이 숫자다."""

    def __post_init__(self) -> None:
        if self.sample_count < 1:
            raise InvariantViolation(
                "표본이 없으면 동등성을 확인한 것이 아니다.", subject="sample_count"
            )
        if self.max_abs_diff < 0 or self.mean_abs_diff < 0:
            raise InvariantViolation("차이는 음수일 수 없다.", subject="diff")
        if not 0.0 <= self.agreement_ratio <= 1.0:
            raise InvariantViolation(
                "agreement_ratio 는 0~1 이어야 한다.", subject="agreement_ratio"
            )

    @property
    def disagreement_count(self) -> int:
        return round(self.sample_count * (1.0 - self.agreement_ratio))

    def describe(self) -> str:
        return (
            f"표본 {self.sample_count}개  max|diff| {self.max_abs_diff:.3e}  "
            f"argmax 일치 {self.agreement_ratio:.2%}"
        )


@dataclass(frozen=True, slots=True)
class ConversionRecord:
    """무엇을 무엇으로 바꿨고, 그 결과가 같은지."""

    source_runtime: RuntimeTarget
    target_runtime: RuntimeTarget
    precision: Precision
    equivalence: NumericalEquivalence | None = None
    note: str = ""

    @property
    def verified(self) -> bool:
        return self.equivalence is not None

    def describe(self) -> str:
        arrow = f"{self.source_runtime.value} → {self.target_runtime.value}"
        if self.equivalence is None:
            return f"{arrow} ({self.precision.value})  ⚠ 동등성 미확인"
        return f"{arrow} ({self.precision.value})  {self.equivalence.describe()}"


@dataclass(frozen=True, slots=True)
class EquivalencePolicy:
    """변환 결과를 믿어도 되는지에 대한 기준.

    정밀도마다 허용 오차가 다르다. FP16 을 FP32 기준으로 재면 전부 실패하고,
    INT8 을 FP16 기준으로 재면 전부 실패한다. 그러면 아무도 이 검사를 쓰지 않는다.
    """

    max_abs_diff_fp32: float = 1e-4
    max_abs_diff_fp16: float = 1e-2
    min_agreement_ratio: float = 0.99
    min_sample_count: int = 32
    """이보다 적은 표본으로 '같다'고 말할 수 없다."""

    def tolerance_for(self, precision: Precision) -> float | None:
        """INT8 은 값 자체를 비교하지 않는다. 애초에 다른 숫자를 쓰기 때문이다."""
        if precision is Precision.FP32:
            return self.max_abs_diff_fp32
        if precision is Precision.FP16:
            return self.max_abs_diff_fp16
        return None

    def inspect(self, record: ConversionRecord) -> tuple[Finding, ...]:
        if record.source_runtime is record.target_runtime and record.equivalence is None:
            # 변환하지 않았다. 기준 모델은 자기 자신과 같은지 확인할 필요가 없다.
            return ()

        if record.equivalence is None:
            return (
                Finding(
                    code="CONVERT_NOT_VERIFIED",
                    message=(
                        "변환 후 같은 답을 내는지 확인하지 않았다. "
                        "변환 도구는 축이 틀려도 성공한다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=record.target_runtime.value,
                ),
            )

        findings: list[Finding] = []
        equivalence = record.equivalence

        if equivalence.sample_count < self.min_sample_count:
            findings.append(
                Finding(
                    code="CONVERT_SAMPLE_TOO_FEW",
                    message=(
                        f"{equivalence.sample_count} 개 표본으로 동등성을 판단했다. "
                        "그 숫자로는 우연히 맞았을 수 있다."
                    ),
                    severity=Severity.WARNING,
                    subject=record.target_runtime.value,
                    measured=float(equivalence.sample_count),
                    threshold=float(self.min_sample_count),
                )
            )

        tolerance = self.tolerance_for(record.precision)
        if tolerance is not None and equivalence.max_abs_diff > tolerance:
            findings.append(
                Finding(
                    code="CONVERT_OUTPUT_DRIFT",
                    message=(
                        f"변환 전후 출력이 {equivalence.max_abs_diff:.3e} 만큼 다르다. "
                        "축 순서나 연산자 치환을 의심한다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=record.target_runtime.value,
                    measured=equivalence.max_abs_diff,
                    threshold=tolerance,
                )
            )

        if equivalence.agreement_ratio < self.min_agreement_ratio:
            findings.append(
                Finding(
                    code="CONVERT_PREDICTION_CHANGED",
                    message=(
                        f"{equivalence.disagreement_count} 개 표본에서 예측이 바뀌었다. "
                        "값이 조금 달라진 것이 아니라 **답이 달라진 것**이다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=record.target_runtime.value,
                    measured=equivalence.agreement_ratio,
                    threshold=self.min_agreement_ratio,
                )
            )

        return tuple(findings)
