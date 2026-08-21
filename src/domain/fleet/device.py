"""수천 개의 디바이스 데이터를 하나로 관리하라. (실습 6-3, 6-11)

디바이스가 3,000대면 **목록은 답이 아니다.** 아무도 3,000줄을 읽지 않는다.
필요한 것은 집계다.

    몇 대가 어느 버전으로 돌고 있는가
    몇 대가 연락이 안 되는가
    몇 대가 격리돼 있는가

그리고 규모가 커지면 두 가지가 늘 참이다.

    1. **몇 대는 언제나 연락이 안 된다.** 꺼져 있거나, 회선이 끊겼거나, 정비 중이다.
    2. **버전은 절대 한 줄로 정렬되지 않는다.** 어제 배포한 것이 아직 다 안 갔다.

두 번째를 version skew 라고 부른다. 이것을 0으로 만들려는 설계는 실패한다.
할 수 있는 것은 **얼마나 벌어져 있는지 아는 것**뿐이다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


class DeviceStatus(Enum):
    HEALTHY = "HEALTHY"
    """제때 보고하고 있다."""

    STALE = "STALE"
    """한동안 연락이 없다. 아직 죽었다고 단정하지 않는다."""

    UNREACHABLE = "UNREACHABLE"
    """오래 연락이 없다. 사람이 가 봐야 한다."""

    QUARANTINED = "QUARANTINED"
    """모듈 5 가 판단을 멈춰 세웠다. **이 디바이스의 데이터는 학습에 쓰지 않는다.**"""

    RETIRED = "RETIRED"


@dataclass(frozen=True, slots=True)
class Device:
    """디바이스 한 대.

    Fleet Aggregate 안의 Entity 다. 밖에서 직접 만들지 않는다.
    """

    device_id: str
    group: str
    current_version: str = ""
    """지금 돌고 있는 릴리스. 비어 있으면 아직 아무것도 못 받았다."""

    last_seen_at: str = ""
    status: DeviceStatus = DeviceStatus.HEALTHY
    site: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if not self.device_id.strip():
            raise InvariantViolation(
                "식별자 없는 디바이스는 현장에서 찾을 수 없다.", subject="device_id"
            )
        if not self.group.strip():
            raise InvariantViolation(
                "그룹이 없으면 단계적 배포를 할 수 없다.", subject="group"
            )

    @property
    def has_reported(self) -> bool:
        return bool(self.last_seen_at)

    @property
    def is_reachable(self) -> bool:
        """지금 새 모델을 내보내면 받을 가능성이 있는가."""
        return self.status in (DeviceStatus.HEALTHY, DeviceStatus.STALE)

    @property
    def is_trainable_source(self) -> bool:
        """이 디바이스의 데이터를 학습에 써도 되는가. (실습 6-4)

        격리된 디바이스는 **이상한 상태에서 낸 판단**을 올린다.
        그것으로 다시 학습하면 이상을 학습한다.
        """
        return self.status in (DeviceStatus.HEALTHY, DeviceStatus.STALE)

    def describe(self) -> str:
        version = self.current_version or "(없음)"
        return (
            f"{self.device_id:<10}{self.group:<12}{version:<18}"
            f"{self.status.value:<14}{self.last_seen_at}"
        )


@dataclass(frozen=True, slots=True)
class FleetSummary:
    """3,000대를 한 화면에.

    **목록이 아니라 집계다.** 이 여섯 줄이 매일 아침 보는 화면이다.
    """

    total: int
    by_status: Mapping[str, int] = field(default_factory=dict)
    by_version: Mapping[str, int] = field(default_factory=dict)
    by_group: Mapping[str, int] = field(default_factory=dict)
    never_reported: int = 0

    @property
    def reachable(self) -> int:
        return self.by_status.get(DeviceStatus.HEALTHY.value, 0) + self.by_status.get(
            DeviceStatus.STALE.value, 0
        )

    @property
    def version_count(self) -> int:
        return len([v for v in self.by_version if v])

    @property
    def dominant_version(self) -> str:
        real = {v: n for v, n in self.by_version.items() if v}
        return max(real, key=lambda v: real[v]) if real else ""

    @property
    def dominant_share(self) -> float:
        if not self.total:
            return 0.0
        return self.by_version.get(self.dominant_version, 0) / self.total

    @property
    def stale_ratio(self) -> float:
        if not self.total:
            return 0.0
        unreachable = self.by_status.get(
            DeviceStatus.UNREACHABLE.value, 0
        ) + self.by_status.get(DeviceStatus.STALE.value, 0)
        return unreachable / self.total

    def render(self) -> str:
        lines = [f"디바이스 {self.total:,}대", "-" * 52]
        lines.append("  상태")
        for name, count in sorted(self.by_status.items(), key=lambda x: -x[1]):
            lines.append(f"    {name:<16}{count:>7,}대  ({count / self.total:>6.1%})")
        lines.append("  버전")
        for name, count in sorted(self.by_version.items(), key=lambda x: -x[1]):
            label = name or "(받은 적 없음)"
            lines.append(f"    {label:<16}{count:>7,}대  ({count / self.total:>6.1%})")
        lines.append("  그룹")
        for name, count in sorted(self.by_group.items(), key=lambda x: -x[1]):
            lines.append(f"    {name:<16}{count:>7,}대")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class FleetHealthPolicy:
    """수천 대를 어디까지 정상이라고 부를 것인가. (실습 6-11)"""

    max_stale_ratio: float = 0.05
    """연락 안 되는 비율. **0 을 요구하는 정책은 매일 실패한다.**"""

    max_version_count: int = 3
    """동시에 돌아도 되는 버전 수. 3개를 넘으면 무엇이 문제인지 못 가린다."""

    min_dominant_share: float = 0.8
    """가장 많은 버전이 이만큼은 돼야 한다. 안 그러면 배포가 멈춰 있는 것이다."""

    max_never_reported: int = 0
    """등록만 되고 한 번도 안 올라온 디바이스. 대개 설치가 안 끝난 것이다."""

    def inspect(self, summary: FleetSummary) -> tuple[Finding, ...]:
        if summary.total == 0:
            return (
                Finding(
                    code="FLEET_EMPTY",
                    message="등록된 디바이스가 없다.",
                    severity=Severity.WARNING,
                    subject="fleet",
                ),
            )

        findings: list[Finding] = []

        if summary.stale_ratio > self.max_stale_ratio:
            findings.append(
                Finding(
                    code="FLEET_TOO_MANY_STALE",
                    message=(
                        f"{summary.stale_ratio:.1%} 가 연락이 안 된다. "
                        "몇 대가 꺼져 있는 것은 정상이지만, 이 비율은 회선이나 "
                        "에이전트 쪽 문제에 가깝다."
                    ),
                    severity=Severity.WARNING,
                    subject="reachability",
                    measured=summary.stale_ratio,
                    threshold=self.max_stale_ratio,
                )
            )

        if summary.version_count > self.max_version_count:
            findings.append(
                Finding(
                    code="FLEET_VERSION_SKEW",
                    message=(
                        f"{summary.version_count}종의 버전이 동시에 돌고 있다. "
                        "문제가 생겨도 **어느 버전의 문제인지 가릴 수 없다.**"
                    ),
                    severity=Severity.CRITICAL,
                    subject="version",
                    measured=float(summary.version_count),
                    threshold=float(self.max_version_count),
                )
            )

        if summary.dominant_share < self.min_dominant_share:
            findings.append(
                Finding(
                    code="FLEET_ROLLOUT_STALLED",
                    message=(
                        f"가장 많은 버전이 {summary.dominant_share:.1%} 뿐이다. "
                        "배포가 중간에 멈춰 있다 — 끝내거나 되돌려야 한다."
                    ),
                    severity=Severity.WARNING,
                    subject="rollout",
                    measured=summary.dominant_share,
                    threshold=self.min_dominant_share,
                )
            )

        if summary.never_reported > self.max_never_reported:
            findings.append(
                Finding(
                    code="FLEET_NEVER_REPORTED",
                    message=(
                        f"{summary.never_reported}대가 등록만 되고 한 번도 올라온 적이 없다. "
                        "대개 설치가 안 끝났거나 자격증명이 안 들어간 것이다."
                    ),
                    severity=Severity.WARNING,
                    subject="onboarding",
                    measured=float(summary.never_reported),
                    threshold=float(self.max_never_reported),
                )
            )

        return tuple(findings)
