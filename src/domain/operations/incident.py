"""문제가 발생하면 모델을 격리하라. (실습 5-8)

격리는 롤백이 아니다. **판단을 멈추는 것**이다.

    격리(quarantine)  이 모델의 판단을 더 이상 쓰지 않는다. 설비는 사람이 본다.
    롤백(rollback)    이전 모델로 되돌린다 (실습 5-10)

격리가 먼저인 이유:
이전 모델이 더 나으리라는 보장이 없다. 입력이 변한 것이라면 이전 모델도 똑같이 틀린다.
**모르는 상태에서 할 수 있는 가장 안전한 일은 멈추는 것이다.**

그리고 격리는 자동이어야 한다.
사람이 알아채고 결정하기까지 걸리는 시간 동안 불량이 계속 나가기 때문이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from domain.operations.health import HealthReport
from domain.operations.identifiers import IncidentId
from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


class IncidentKind(Enum):
    LATENCY = "LATENCY"
    PREDICTION_SHIFT = "PREDICTION_SHIFT"
    INPUT_DRIFT = "INPUT_DRIFT"
    LOGGING = "LOGGING"
    MIXED = "MIXED"


class IncidentStatus(Enum):
    OPEN = "OPEN"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


@dataclass(frozen=True, slots=True)
class Incident:
    """현장에서 실제로 벌어진 사건 하나."""

    incident_id: IncidentId
    kind: IncidentKind
    opened_at: str
    window_label: str
    deployment_version: int
    findings: tuple[Finding, ...] = field(default_factory=tuple)
    status: IncidentStatus = IncidentStatus.OPEN
    resolution: str = ""

    def __post_init__(self) -> None:
        if not self.findings:
            raise InvariantViolation(
                "근거 없는 사건은 기록하지 않는다.", subject="findings"
            )
        if not self.opened_at.strip():
            raise InvariantViolation("사건 발생 시각이 없다.", subject="opened_at")

    @property
    def is_open(self) -> bool:
        return self.status is not IncidentStatus.RESOLVED

    @property
    def summary(self) -> str:
        codes = ", ".join(sorted({f.code for f in self.findings}))
        return f"[{self.kind.value}] {self.window_label} v{self.deployment_version} — {codes}"

    def resolved(self, resolution: str) -> Incident:
        if not resolution.strip():
            raise InvariantViolation(
                "무엇을 해서 끝났는지 없으면 같은 일이 또 일어난다.",
                subject="resolution",
            )
        return Incident(
            incident_id=self.incident_id,
            kind=self.kind,
            opened_at=self.opened_at,
            window_label=self.window_label,
            deployment_version=self.deployment_version,
            findings=self.findings,
            status=IncidentStatus.RESOLVED,
            resolution=resolution.strip(),
        )

    def render(self) -> str:
        lines = [f"{self.summary}  ({self.status.value})"]
        lines += [f"    {f.describe()}" for f in self.findings]
        if self.resolution:
            lines.append(f"    → {self.resolution}")
        return "\n".join(lines)


_KIND_BY_PREFIX: tuple[tuple[str, IncidentKind], ...] = (
    ("OPS_OVER_CYCLE_BUDGET", IncidentKind.LATENCY),
    ("OPS_LATENCY", IncidentKind.LATENCY),
    ("OPS_TIMEOUT", IncidentKind.LATENCY),
    ("OPS_PREDICTION", IncidentKind.PREDICTION_SHIFT),
    ("OPS_LABEL", IncidentKind.PREDICTION_SHIFT),
    ("OPS_CONFIDENCE", IncidentKind.PREDICTION_SHIFT),
    ("OPS_LOW_CONFIDENCE", IncidentKind.PREDICTION_SHIFT),
    ("OPS_INPUT", IncidentKind.INPUT_DRIFT),
    ("OPS_MULTI_FEATURE", IncidentKind.INPUT_DRIFT),
    ("LOG_", IncidentKind.LOGGING),
)


@dataclass(frozen=True, slots=True)
class IncidentPolicy:
    """무엇을 사건으로 부르고, 언제 판단을 멈출 것인가."""

    quarantine_on_critical: bool = True
    """CRITICAL 이 하나라도 나오면 자동으로 격리한다."""

    quarantine_on_warning_count: int = 4
    """WARNING 이 이만큼 쌓여도 격리한다. 작은 이상이 여럿이면 큰 이상이다."""

    def assess(self, report: HealthReport) -> tuple[IncidentKind, ...] | None:
        """사건인가? 사건이면 어떤 종류인가."""
        if not report.findings:
            return None
        kinds = {self._kind_of(f) for f in report.findings}
        return tuple(sorted(kinds, key=lambda k: k.value))

    def should_quarantine(self, report: HealthReport) -> tuple[bool, str]:
        """판단을 멈춰야 하는가. 멈춰야 한다면 그 이유도 함께 돌려준다."""
        critical = [f for f in report.findings if f.severity is Severity.CRITICAL]
        if self.quarantine_on_critical and critical:
            reasons = "; ".join(f.describe() for f in critical)
            return True, f"{report.window.label}: {reasons}"

        warnings = [f for f in report.findings if f.severity is Severity.WARNING]
        if len(warnings) >= self.quarantine_on_warning_count:
            reasons = "; ".join(f.code for f in warnings)
            return True, (
                f"{report.window.label}: 경고가 {len(warnings)}건 겹쳤다 ({reasons}). "
                "작은 이상이 여럿이면 큰 이상이다."
            )
        return False, ""

    def kind_of(self, report: HealthReport) -> IncidentKind:
        kinds = self.assess(report)
        if not kinds:
            return IncidentKind.MIXED
        return kinds[0] if len(kinds) == 1 else IncidentKind.MIXED

    def _kind_of(self, finding: Finding) -> IncidentKind:
        for prefix, kind in _KIND_BY_PREFIX:
            if finding.code.startswith(prefix):
                return kind
        return IncidentKind.MIXED
