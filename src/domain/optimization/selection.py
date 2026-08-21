"""가장 빠른 모델이 아니라 가장 쓸 수 있는 모델을 선택하라. (실습 4-10)

선택 규칙은 한 줄이다.

    **예산을 만족하는 후보 중에서 가장 정확한 것.**

'가장 빠른 것'이 아니다. 예산을 이미 만족했다면 그보다 더 빠른 것은
아무 값어치가 없고, 그 속도를 얻으려고 내준 정확도만 손해다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from domain.optimization.conversion import EquivalencePolicy
from domain.optimization.identifiers import ArtifactId
from domain.optimization.runtime import RuntimeTarget
from domain.optimization.tradeoff import OptimizationCandidate, TradeoffTable
from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity, Verdict


@dataclass(frozen=True, slots=True)
class DeviceBudget:
    """디바이스가 허용하는 것.

    이 숫자들은 모델이 정하지 않는다. **설비와 하드웨어가 정한다.**
    사이클 타임 30ms, 플래시 256KiB, RAM 64KiB — 그런 식이다.
    """

    name: str
    latency_p95_ms: float
    storage_kib: float | None = None
    activation_kib: float | None = None
    min_macro_recall: float = 0.0
    max_accuracy_drop: float = 0.02
    max_class_recall_drop: float = 0.05
    """어느 한 클래스라도 이만큼 떨어지면 안 된다. **평균은 이것을 숨긴다.**"""

    def __post_init__(self) -> None:
        if self.latency_p95_ms <= 0:
            raise InvariantViolation(
                "지연시간 예산은 0보다 커야 한다.", subject="latency_p95_ms"
            )
        for name in ("storage_kib", "activation_kib"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise InvariantViolation(f"{name} 는 0보다 커야 한다.", subject=name)
        for name in ("min_macro_recall", "max_accuracy_drop", "max_class_recall_drop"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise InvariantViolation(f"{name} 는 0~1 이어야 한다.", subject=name)

    def describe(self) -> str:
        parts = [f"p95 ≤ {self.latency_p95_ms:g}ms"]
        if self.storage_kib:
            parts.append(f"저장 ≤ {self.storage_kib:g}KiB")
        if self.activation_kib:
            parts.append(f"활성 ≤ {self.activation_kib:g}KiB")
        parts.append(f"macro recall ≥ {self.min_macro_recall:.2f}")
        parts.append(f"정확도 하락 ≤ {self.max_accuracy_drop:.2%}")
        return f"{self.name}: " + ", ".join(parts)


class SelectionObjective(Enum):
    ACCURACY = "ACCURACY"
    """예산 안에서 가장 정확한 것. **기본값이자 대개 옳은 선택.**"""

    LATENCY = "LATENCY"
    """예산 안에서 가장 빠른 것. 여유를 남겨야 할 이유가 있을 때만."""

    SIZE = "SIZE"
    """예산 안에서 가장 작은 것. 플래시가 정말 모자랄 때."""


@dataclass(frozen=True, slots=True)
class CandidateVerdict:
    """후보 하나에 대한 판정과 그 근거."""

    label: str
    accepted: bool
    findings: tuple[Finding, ...] = field(default_factory=tuple)

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(f.describe() for f in self.findings)


@dataclass(frozen=True, slots=True)
class OptimizationCertificate:
    """선택의 결과이자 근거 기록."""

    budget: DeviceBudget
    objective: SelectionObjective
    verdict: Verdict
    selected_artifact_id: ArtifactId | None
    selected_label: str | None
    selected_p95_ms: float | None
    selected_accuracy: float | None
    verdicts: tuple[CandidateVerdict, ...] = field(default_factory=tuple)
    table: str = ""

    @property
    def has_selection(self) -> bool:
        return self.selected_artifact_id is not None

    def render(self) -> str:
        lines = [
            f"모델 선택: {self.verdict.value}  ({self.budget.name})",
            f"  예산 : {self.budget.describe()}",
            f"  목표 : {self.objective.value}",
        ]
        if self.has_selection:
            lines.append(
                f"  선택 : {self.selected_label}  "
                f"p95 {self.selected_p95_ms:.4f}ms  정확도 {self.selected_accuracy:.4f}"
            )
        else:
            lines.append("  선택 : 없음 — 예산을 만족하는 후보가 하나도 없다")
        if self.table:
            lines.append("")
            lines.append(self.table)
        rejected = [v for v in self.verdicts if not v.accepted]
        if rejected:
            lines.append("")
            lines.append("탈락 사유:")
            for candidate in rejected:
                lines.append(f"  ✗ {candidate.label}")
                lines += [f"      {reason}" for reason in candidate.reasons]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    """무엇을 '쓸 수 있는 모델'이라고 부를 것인가."""

    budget: DeviceBudget
    objective: SelectionObjective = SelectionObjective.ACCURACY
    equivalence: EquivalencePolicy = field(default_factory=EquivalencePolicy)
    require_deployable_runtime: bool = True
    """PyTorch 는 파이썬을 요구한다. 디바이스에 파이썬은 대개 없다."""

    latency_tie_ratio: float = 0.05
    """이만큼 안쪽의 속도 차이는 **차이가 아니다.**

    0.0030ms 와 0.0031ms 중에 고르는 것은 측정 잡음 중에 고르는 것이다.
    다시 재면 순서가 뒤집힌다 — 그러면 같은 입력에 다른 답을 내는 선택이 된다.
    잡음 안쪽에서는 속도로 고르지 않고, 그 다음 기준(크기)으로 넘어간다.
    """

    def evaluate(self, table: TradeoffTable) -> OptimizationCertificate:
        verdicts: list[CandidateVerdict] = []
        accepted: list[OptimizationCandidate] = []

        for candidate in table.all_candidates:
            findings = self._check(candidate, table.baseline)
            is_accepted = not any(f.severity is Severity.CRITICAL for f in findings)
            verdicts.append(
                CandidateVerdict(
                    label=candidate.label, accepted=is_accepted, findings=findings
                )
            )
            if is_accepted:
                accepted.append(candidate)

        selected = self._pick(accepted)
        return OptimizationCertificate(
            budget=self.budget,
            objective=self.objective,
            verdict=Verdict.PASSED if selected else Verdict.FAILED,
            selected_artifact_id=selected.artifact.artifact_id if selected else None,
            selected_label=selected.label if selected else None,
            selected_p95_ms=selected.benchmark.p95_ms if selected else None,
            selected_accuracy=selected.accuracy.accuracy if selected else None,
            verdicts=tuple(verdicts),
            table=table.render(),
        )

    # -- 내부 --------------------------------------------------------------
    def _pick(
        self, accepted: list[OptimizationCandidate]
    ) -> OptimizationCandidate | None:
        if not accepted:
            return None
        if self.objective is SelectionObjective.LATENCY:
            return min(accepted, key=lambda c: c.benchmark.p95_ms)
        if self.objective is SelectionObjective.SIZE:
            return min(accepted, key=lambda c: c.artifact.size_bytes)

        # 예산을 이미 만족했다면 남은 여유는 값어치가 없다. 정확도를 고른다.
        best = max(
            accepted, key=lambda c: (c.accuracy.macro_recall, c.accuracy.accuracy)
        )
        tied = [
            c
            for c in accepted
            if (c.accuracy.macro_recall, c.accuracy.accuracy)
            == (best.accuracy.macro_recall, best.accuracy.accuracy)
        ]
        if len(tied) == 1:
            return tied[0]

        # 정확도가 같다면 속도를 본다. 단 **잡음 수준의 차이는 무시한다.**
        fastest = min(c.benchmark.p95_ms for c in tied)
        contenders = [
            c
            for c in tied
            if c.benchmark.p95_ms <= fastest * (1.0 + self.latency_tie_ratio)
        ]
        # 정확도도 속도도 사실상 같다면 작은 것을 고른다. 플래시는 언제나 아쉽다.
        return min(contenders, key=lambda c: (c.artifact.size_bytes, c.label))

    def _check(
        self, candidate: OptimizationCandidate, baseline: OptimizationCandidate
    ) -> tuple[Finding, ...]:
        findings: list[Finding] = []
        budget = self.budget

        if (
            self.require_deployable_runtime
            and not candidate.artifact.runtime.is_deployable_on_device
        ):
            findings.append(
                Finding(
                    code="SELECT_RUNTIME_NOT_DEPLOYABLE",
                    message=(
                        f"{candidate.artifact.runtime.value} 는 파이썬 런타임을 요구한다. "
                        "디바이스에 파이썬이 있는지부터 확인해야 한다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=candidate.label,
                )
            )

        if candidate.benchmark.p95_ms > budget.latency_p95_ms:
            findings.append(
                Finding(
                    code="SELECT_OVER_LATENCY_BUDGET",
                    message="사이클 타임 안에 못 들어온다.",
                    severity=Severity.CRITICAL,
                    subject=candidate.label,
                    measured=candidate.benchmark.p95_ms,
                    threshold=budget.latency_p95_ms,
                )
            )

        if budget.storage_kib is not None and candidate.artifact.size_kib > budget.storage_kib:
            findings.append(
                Finding(
                    code="SELECT_OVER_STORAGE_BUDGET",
                    message="플래시에 안 들어간다.",
                    severity=Severity.CRITICAL,
                    subject=candidate.label,
                    measured=candidate.artifact.size_kib,
                    threshold=budget.storage_kib,
                )
            )

        if budget.activation_kib is not None:
            activation_kib = candidate.benchmark.activation_bytes / 1024
            if activation_kib > budget.activation_kib:
                findings.append(
                    Finding(
                        code="SELECT_OVER_ACTIVATION_BUDGET",
                        message=(
                            "중간 결과가 RAM 을 넘는다. "
                            "모델 파일이 작아도 이쪽에서 막히는 일이 잦다."
                        ),
                        severity=Severity.CRITICAL,
                        subject=candidate.label,
                        measured=activation_kib,
                        threshold=budget.activation_kib,
                    )
                )

        if candidate.accuracy.macro_recall < budget.min_macro_recall:
            findings.append(
                Finding(
                    code="SELECT_BELOW_MACRO_RECALL",
                    message="클래스 평균 재현율이 기준 아래다.",
                    severity=Severity.CRITICAL,
                    subject=candidate.label,
                    measured=candidate.accuracy.macro_recall,
                    threshold=budget.min_macro_recall,
                )
            )

        drop = candidate.accuracy.drop_from(baseline.accuracy)
        if drop > budget.max_accuracy_drop:
            findings.append(
                Finding(
                    code="SELECT_ACCURACY_DROP",
                    message="기준 모델 대비 정확도가 너무 떨어졌다.",
                    severity=Severity.CRITICAL,
                    subject=candidate.label,
                    measured=drop,
                    threshold=budget.max_accuracy_drop,
                )
            )

        label, class_drop = candidate.accuracy.worst_class_drop_from(baseline.accuracy)
        if label is not None and class_drop > budget.max_class_recall_drop:
            findings.append(
                Finding(
                    code="SELECT_CLASS_RECALL_DROP",
                    message=(
                        f"'{label}' 재현율이 {class_drop:.2%} 떨어졌다. "
                        "전체 정확도는 거의 그대로인데 한 클래스만 무너진 것이다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=candidate.label,
                    measured=class_drop,
                    threshold=budget.max_class_recall_drop,
                )
            )

        findings.extend(self.equivalence.inspect(candidate.conversion))

        if candidate.benchmark.p95_ms > baseline.benchmark.p95_ms:
            findings.append(
                Finding(
                    code="SELECT_SLOWER_THAN_BASELINE",
                    message=(
                        "최적화했는데 기준 모델보다 느리다. "
                        "양자화한 값을 다시 풀어 쓰는 비용이 계산 이득보다 클 때 그렇다."
                    ),
                    severity=Severity.WARNING,
                    subject=candidate.label,
                    measured=candidate.benchmark.p95_ms,
                    threshold=baseline.benchmark.p95_ms,
                )
            )

        return tuple(findings)


DEFAULT_RUNTIME_ORDER: tuple[RuntimeTarget, ...] = (
    RuntimeTarget.PYTORCH,
    RuntimeTarget.TORCHSCRIPT,
    RuntimeTarget.ONNX,
    RuntimeTarget.TFLITE,
)
