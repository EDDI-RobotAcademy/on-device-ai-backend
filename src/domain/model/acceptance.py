"""현장 데이터를 통과하는 모델만 살아남는다. (실습 3-10)

학습이 끝났다고 모델이 완성된 것이 아니다.
"검증 정확도가 제일 높은 모델"과 "현장에 내보낼 수 있는 모델"은 다르다.

여기서 묻는 것은 넷이다.

    1. 학습이 실제로 일어났는가        (3-6)
    2. 외운 것이 아닌가                (3-7)
    3. 정확도 뒤에 실패가 없는가        (3-9)
    4. 디바이스에서 시간 안에 도는가    → 모듈 4로 이어진다
"""

from __future__ import annotations

from dataclasses import dataclass, field

from domain.model.curve import (
    LearningPolicy,
    OverfittingPolicy,
    TrainingCurve,
)
from domain.model.evaluation import EvaluationPolicy, EvaluationResult
from domain.model.identifiers import ModelVersionId
from domain.model.protocol import EvaluationProtocol, SplitUsage
from domain.shared.errors import InvariantViolation
from domain.shared.inspection import (
    Finding,
    Severity,
    Verdict,
    blocking_findings,
    derive_verdict,
    warning_findings,
)


@dataclass(frozen=True, slots=True)
class LatencyBudget:
    """디바이스에서 허용되는 시간.

    이 숫자는 모델이 정하는 것이 아니다. **현장의 사이클 타임이 정한다.**
    30ms 라면, 그것은 설비가 30ms 마다 판단을 요구한다는 뜻이다.
    """

    p95_ms: float
    memory_mib: float | None = None

    def __post_init__(self) -> None:
        if self.p95_ms <= 0:
            raise InvariantViolation("지연시간 예산은 0보다 커야 한다.", subject="p95_ms")
        if self.memory_mib is not None and self.memory_mib <= 0:
            raise InvariantViolation("메모리 예산은 0보다 커야 한다.", subject="memory_mib")


@dataclass(frozen=True, slots=True)
class ModelCertificate:
    """배포 가능 판정의 결과이자 근거 기록."""

    model_version_id: ModelVersionId
    verdict: Verdict
    accuracy: float
    macro_recall: float
    latency_ms_p95: float
    blocking: tuple[Finding, ...] = field(default_factory=tuple)
    warnings: tuple[Finding, ...] = field(default_factory=tuple)

    @property
    def is_deployable(self) -> bool:
        return self.verdict is not Verdict.FAILED

    def render(self) -> str:
        lines = [
            f"모델 승인 판정: {self.verdict.value}  ({self.model_version_id})",
            f"  정확도 {self.accuracy:.3f} / macro recall {self.macro_recall:.3f} "
            f"/ p95 {self.latency_ms_p95:.2f}ms",
        ]
        if self.blocking:
            lines.append("")
            lines.append("차단 사유:")
            lines += [f"  ✗ {f.describe()}" for f in self.blocking]
        if self.warnings:
            lines.append("")
            lines.append("경고(배포는 가능):")
            lines += [f"  ! {f.describe()}" for f in self.warnings]
        if self.is_deployable and not self.warnings:
            lines.append("")
            lines.append("  ✓ 막는 것도, 걸리는 것도 없다.")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ModelAcceptancePolicy:
    """무엇을 '현장에 내보낼 수 있는 모델'이라고 부를 것인가."""

    learning: LearningPolicy = field(default_factory=LearningPolicy)
    overfitting: OverfittingPolicy = field(default_factory=OverfittingPolicy)
    evaluation: EvaluationPolicy = field(default_factory=EvaluationPolicy)
    protocol: EvaluationProtocol = field(default_factory=EvaluationProtocol)
    latency: LatencyBudget | None = None
    require_gates: bool = True
    """모듈 1·2 게이트를 통과한 데이터로 학습했는가."""

    def evaluate(
        self,
        model_version_id: ModelVersionId,
        *,
        curve: TrainingCurve,
        result: EvaluationResult,
        usage: SplitUsage,
        baseline_accuracy: float = 0.0,
        gates_passed: bool = True,
        missing_gates: tuple[str, ...] = (),
    ) -> ModelCertificate:
        findings: list[Finding] = []

        if self.require_gates and not gates_passed:
            findings.append(
                Finding(
                    code="ACCEPT_GATES_NOT_PASSED",
                    message=(
                        "게이트를 통과하지 않은 데이터로 학습했다: "
                        + ", ".join(missing_gates)
                    ),
                    severity=Severity.CRITICAL,
                    subject="data",
                )
            )

        findings.extend(self.learning.inspect(curve, baseline_accuracy))
        findings.extend(self.overfitting.inspect(curve))
        findings.extend(self.protocol.inspect(usage))
        findings.extend(self.evaluation.inspect(result))

        if self.latency is not None:
            if result.latency_ms_p95 > self.latency.p95_ms:
                findings.append(
                    Finding(
                        code="ACCEPT_LATENCY_OVER_BUDGET",
                        message=(
                            f"p95 지연시간 {result.latency_ms_p95:.2f}ms 가 "
                            f"예산 {self.latency.p95_ms:.2f}ms 를 넘는다. "
                            "정확도와 무관하게 이 모델은 현장에서 못 쓴다."
                        ),
                        severity=Severity.CRITICAL,
                        subject="latency",
                        measured=result.latency_ms_p95,
                        threshold=self.latency.p95_ms,
                    )
                )

        matrix = result.matrix
        return ModelCertificate(
            model_version_id=model_version_id,
            verdict=derive_verdict(tuple(findings)),
            accuracy=matrix.accuracy,
            macro_recall=matrix.macro_recall,
            latency_ms_p95=result.latency_ms_p95,
            blocking=blocking_findings(tuple(findings)),
            warnings=warning_findings(tuple(findings)),
        )
