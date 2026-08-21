"""OTA로 현장의 AI를 업데이트하라. (실습 6-8, 6-9)

모듈 5 의 배포와 결정적으로 다른 점이 둘 있다.

    1. **한 번에 다 안 간다.**
       3,000대 중 몇 대는 꺼져 있고, 몇 대는 회선이 끊겼다.
       "배포 완료"는 100% 가 아니라 **"받을 수 있는 것은 다 받았다"** 이다.

    2. **되돌리는 것도 즉시 안 된다.**
       모듈 5 의 롤백은 기록 하나를 바꾸는 일이었다.
       여기서는 롤백도 **또 하나의 롤아웃**이고, 똑같이 시간이 걸린다.

그래서 단계(wave)로 나눈다.

    wave 1   canary 몇 대       → 실패율을 본다
    wave 2   한 그룹            → 실패율을 본다
    wave 3   전부

각 단계 뒤에 멈출 수 있어야 의미가 있다.
멈추지 않을 거면 단계로 나눌 이유가 없다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from domain.fleet import events as domain_events
from domain.fleet.errors import RolloutHalted
from domain.fleet.identifiers import RolloutId
from domain.shared.errors import IllegalStateTransition, InvariantViolation
from domain.shared.events import EventRecorder
from domain.shared.inspection import Finding, Severity


class DeviceOutcome(Enum):
    PENDING = "PENDING"
    """아직 안 받았다. 꺼져 있을 수도, 그냥 늦을 수도 있다."""

    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    """받았는데 설치나 검증에서 실패했다."""

    UNREACHABLE = "UNREACHABLE"
    """연락 자체가 안 됐다. **실패와 다르다** — 모델 문제가 아니다."""

    SKIPPED = "SKIPPED"
    """대상에서 뺐다. 격리 중이거나 정비 중이다."""


class RolloutStatus(Enum):
    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    HALTED = "HALTED"
    """실패가 기준을 넘어 스스로 멈췄다. **사람이 결정하기 전에 멈춘다.**"""

    COMPLETED = "COMPLETED"
    ROLLED_BACK = "ROLLED_BACK"


@dataclass(frozen=True, slots=True)
class Wave:
    """한 단계. 몇 대에, 어느 그룹에."""

    name: str
    device_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise InvariantViolation("단계에 이름이 없다.", subject="name")
        if not self.device_ids:
            raise InvariantViolation(
                "대상이 0대인 단계는 없다.", subject="device_ids"
            )

    @property
    def size(self) -> int:
        return len(self.device_ids)


@dataclass(frozen=True, slots=True)
class WaveResult:
    """한 단계의 결과."""

    wave: Wave
    outcomes: dict[str, DeviceOutcome] = field(default_factory=dict)

    def count(self, outcome: DeviceOutcome) -> int:
        return sum(1 for value in self.outcomes.values() if value is outcome)

    @property
    def reported(self) -> int:
        """응답이 온 대수. PENDING 은 아직 아무 말이 없다."""
        return sum(
            1 for value in self.outcomes.values() if value is not DeviceOutcome.PENDING
        )

    @property
    def attempted(self) -> int:
        """실제로 시도가 이뤄진 대수. **연락 안 된 것은 시도가 아니다.**"""
        return self.count(DeviceOutcome.SUCCEEDED) + self.count(DeviceOutcome.FAILED)

    @property
    def failure_ratio(self) -> float:
        """시도한 것 중 실패한 비율.

        **분모가 전체가 아니라 '시도한 것'이다.**
        꺼져 있는 디바이스를 실패로 세면 실패율이 늘 높게 나오고,
        그러면 아무 배포도 통과하지 못한다.
        """
        return self.count(DeviceOutcome.FAILED) / self.attempted if self.attempted else 0.0

    @property
    def success_ratio(self) -> float:
        return (
            self.count(DeviceOutcome.SUCCEEDED) / self.wave.size if self.wave.size else 0.0
        )

    @property
    def is_settled(self) -> bool:
        """더 기다릴 이유가 있는가."""
        return self.count(DeviceOutcome.PENDING) == 0

    def describe(self) -> str:
        return (
            f"{self.wave.name:<12}{self.wave.size:>6}대  "
            f"성공 {self.count(DeviceOutcome.SUCCEEDED):>5}  "
            f"실패 {self.count(DeviceOutcome.FAILED):>4}  "
            f"미도달 {self.count(DeviceOutcome.UNREACHABLE):>4}  "
            f"대기 {self.count(DeviceOutcome.PENDING):>4}  "
            f"(실패율 {self.failure_ratio:.1%})"
        )


@dataclass(frozen=True, slots=True)
class RolloutPolicy:
    """어디까지 실패해도 계속할 것인가."""

    max_failure_ratio: float = 0.05
    """시도한 것 중 실패 비율의 상한. 넘으면 **자동으로 멈춘다.**"""

    min_reported_before_advance: float = 0.7
    """다음 단계로 가기 전에 이만큼은 응답이 와야 한다.
    응답 없이 넘어가면 단계를 나눈 의미가 없다."""

    max_unreachable_ratio: float = 0.2
    """연락 안 되는 비율의 상한. 이건 모델 문제가 아니라 회선/전원 문제다."""

    require_canary_first: bool = True
    min_canary_devices: int = 2
    """작은 플릿에서도 이만큼은 첫 단계로 허용한다.

    10% 로만 재면 8대짜리 플릿의 첫 단계가 0대가 된다 — 그건 계획이 아니다.
    """

    def inspect(self, result: WaveResult) -> tuple[Finding, ...]:
        findings: list[Finding] = []
        size = result.wave.size

        if result.failure_ratio > self.max_failure_ratio:
            findings.append(
                Finding(
                    code="OTA_FAILURE_RATE",
                    message=(
                        f"'{result.wave.name}' 에서 시도한 것 중 "
                        f"{result.failure_ratio:.1%} 가 실패했다. "
                        "다음 단계로 넘기지 않는다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=result.wave.name,
                    measured=result.failure_ratio,
                    threshold=self.max_failure_ratio,
                )
            )

        unreachable = result.count(DeviceOutcome.UNREACHABLE) / size if size else 0.0
        if unreachable > self.max_unreachable_ratio:
            findings.append(
                Finding(
                    code="OTA_TOO_MANY_UNREACHABLE",
                    message=(
                        f"{unreachable:.1%} 가 연락이 안 된다. "
                        "**이건 모델 문제가 아니다** — 회선이나 전원 쪽을 먼저 본다."
                    ),
                    severity=Severity.WARNING,
                    subject=result.wave.name,
                    measured=unreachable,
                    threshold=self.max_unreachable_ratio,
                )
            )

        reported = result.reported / size if size else 0.0
        if reported < self.min_reported_before_advance:
            findings.append(
                Finding(
                    code="OTA_NOT_ENOUGH_REPORTED",
                    message=(
                        f"{reported:.1%} 만 응답했다. "
                        "지금 다음 단계로 넘기면 단계를 나눈 의미가 없다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=result.wave.name,
                    measured=reported,
                    threshold=self.min_reported_before_advance,
                )
            )

        return tuple(findings)

    def should_halt(self, result: WaveResult) -> tuple[bool, str]:
        blocking = [
            f
            for f in self.inspect(result)
            if f.severity is Severity.CRITICAL and f.code == "OTA_FAILURE_RATE"
        ]
        if blocking:
            return True, blocking[0].describe()
        return False, ""


class Rollout(EventRecorder):
    """한 번의 OTA 배포. (Aggregate Root)

    지키는 불변식:
        - 단계는 순서대로만 진행한다
        - 멈춘 뒤에는 다음 단계로 못 간다 — 되돌리거나 이유를 붙여 재개한다
        - **롤백도 롤아웃이다** — 즉시 끝나지 않는다
    """

    __slots__ = (
        "_id",
        "_version",
        "_previous_version",
        "_waves",
        "_results",
        "_status",
        "_current",
        "_halt_reason",
        "_history",
    )

    def __init__(
        self,
        rollout_id: RolloutId,
        version: str,
        waves: tuple[Wave, ...],
        previous_version: str = "",
    ) -> None:
        super().__init__()
        self._id = rollout_id
        self._version = version
        self._previous_version = previous_version
        self._waves = waves
        self._results: list[WaveResult] = []
        self._status = RolloutStatus.PLANNED
        self._current = 0
        self._halt_reason = ""
        self._history: list[tuple[str, str]] = []

    # -- 생성 --------------------------------------------------------------
    @classmethod
    def plan(
        cls,
        rollout_id: RolloutId,
        version: str,
        waves: tuple[Wave, ...],
        *,
        previous_version: str = "",
        policy: RolloutPolicy | None = None,
    ) -> Rollout:
        """단계를 짠다. (실습 6-8)

        단계가 커지는 순서여야 한다. 처음부터 크면 나눈 의미가 없다.
        """
        if not waves:
            raise InvariantViolation(
                "단계가 없는 롤아웃은 그냥 전부에게 던지는 것이다.", subject="waves"
            )
        sizes = [wave.size for wave in waves]
        if sizes != sorted(sizes):
            raise InvariantViolation(
                "단계가 커지는 순서가 아니다. 큰 단계를 먼저 하면 "
                "작은 단계에서 확인할 것이 없다.",
                subject="waves",
            )
        rules = policy or RolloutPolicy()
        # 10% 로만 재면 작은 플릿에서는 상한이 0이 된다.
        # 8대짜리 롤백에서 "첫 단계는 0대" 를 요구하는 것은 규칙이 아니라 버그다.
        canary_limit = max(rules.min_canary_devices, sum(sizes) // 10)
        if rules.require_canary_first and sizes[0] > canary_limit:
            raise InvariantViolation(
                f"첫 단계가 {sizes[0]}대다. {canary_limit}대 이하로 시작한다 "
                f"(전체 {sum(sizes)}대의 10% 또는 최소 {rules.min_canary_devices}대) — "
                "문제가 나도 그만큼만 겪는다.",
                subject="waves",
            )

        seen: set[str] = set()
        for wave in waves:
            duplicated = seen & set(wave.device_ids)
            if duplicated:
                raise InvariantViolation(
                    f"{sorted(duplicated)} 가 두 단계에 들어 있다. "
                    "같은 디바이스를 두 번 세면 실패율이 거짓말을 한다.",
                    subject=wave.name,
                )
            seen |= set(wave.device_ids)

        rollout = cls(rollout_id, version, waves, previous_version)
        rollout._record(
            domain_events.RolloutPlanned(
                rollout_id=rollout_id,
                version=version,
                wave_count=len(waves),
                device_count=sum(sizes),
            )
        )
        return rollout

    # -- 조회 --------------------------------------------------------------
    @property
    def id(self) -> RolloutId:
        return self._id

    @property
    def version(self) -> str:
        return self._version

    @property
    def previous_version(self) -> str:
        return self._previous_version

    @property
    def status(self) -> RolloutStatus:
        return self._status

    @property
    def waves(self) -> tuple[Wave, ...]:
        return self._waves

    @property
    def results(self) -> tuple[WaveResult, ...]:
        return tuple(self._results)

    @property
    def current_wave(self) -> Wave | None:
        return self._waves[self._current] if self._current < len(self._waves) else None

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    @property
    def history(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._history)

    @property
    def device_count(self) -> int:
        return sum(wave.size for wave in self._waves)

    @property
    def succeeded_count(self) -> int:
        return sum(r.count(DeviceOutcome.SUCCEEDED) for r in self._results)

    @property
    def failed_count(self) -> int:
        return sum(r.count(DeviceOutcome.FAILED) for r in self._results)

    @property
    def unreachable_count(self) -> int:
        return sum(r.count(DeviceOutcome.UNREACHABLE) for r in self._results)

    @property
    def coverage(self) -> float:
        """전체 중 실제로 새 버전이 올라간 비율.

        **100% 는 오지 않는다.** 꺼진 디바이스가 늘 있다.
        """
        return self.succeeded_count / self.device_count if self.device_count else 0.0

    def render(self) -> str:
        lines = [
            f"롤아웃 {self._id}  {self._version}  → {self._status.value}",
            "-" * 78,
        ]
        lines += [f"  {result.describe()}" for result in self._results]
        pending = self._waves[len(self._results) :]
        lines += [f"  {wave.name:<12}{wave.size:>6}대  (대기)" for wave in pending]
        lines.append("-" * 78)
        lines.append(
            f"  도달 {self.succeeded_count:,}/{self.device_count:,}대 "
            f"({self.coverage:.1%})  실패 {self.failed_count}  "
            f"미도달 {self.unreachable_count}"
        )
        if self._halt_reason:
            lines.append(f"  중단 사유: {self._halt_reason}")
        return "\n".join(lines)

    # -- 행위 --------------------------------------------------------------
    def start(self, occurred_at: str) -> Wave:
        if self._status is not RolloutStatus.PLANNED:
            raise IllegalStateTransition(
                "이미 시작했다.", subject=str(self._id)
            )
        self._status = RolloutStatus.RUNNING
        wave = self._waves[0]
        self._log(occurred_at, f"시작 — {wave.name} ({wave.size}대)")
        self._record(
            domain_events.WaveStarted(
                rollout_id=self._id, wave=wave.name, device_count=wave.size
            )
        )
        return wave

    def record_wave(
        self,
        outcomes: dict[str, DeviceOutcome],
        policy: RolloutPolicy,
        occurred_at: str,
    ) -> WaveResult:
        """지금 단계의 결과를 받아 적는다.

        실패가 기준을 넘으면 **여기서 스스로 멈춘다.**
        사람이 대시보드를 보고 결정하기까지 기다리지 않는다.
        """
        if self._status is not RolloutStatus.RUNNING:
            raise IllegalStateTransition(
                f"{self._status.value} 상태에서는 결과를 기록할 수 없다.",
                subject=str(self._id),
            )
        wave = self.current_wave
        if wave is None:
            raise IllegalStateTransition(
                "남은 단계가 없다.", subject=str(self._id)
            )

        unknown = set(outcomes) - set(wave.device_ids)
        if unknown:
            raise InvariantViolation(
                f"{sorted(unknown)} 는 이 단계의 대상이 아니다.", subject=wave.name
            )

        filled = {device: DeviceOutcome.PENDING for device in wave.device_ids}
        filled.update(outcomes)
        result = WaveResult(wave=wave, outcomes=filled)
        self._results.append(result)
        self._log(occurred_at, result.describe())

        should_halt, reason = policy.should_halt(result)
        if should_halt:
            self._status = RolloutStatus.HALTED
            self._halt_reason = reason
            self._log(occurred_at, f"자동 중단 — {reason}")
            self._record(
                domain_events.RolloutHaltedEvent(
                    rollout_id=self._id, wave=wave.name, reason=reason
                )
            )
        return result

    def advance(self, policy: RolloutPolicy, occurred_at: str) -> Wave:
        """다음 단계로 넘어간다. (실습 6-8)"""
        if self._status is RolloutStatus.HALTED:
            raise RolloutHalted(
                f"멈춘 롤아웃이다: {self._halt_reason}. "
                "되돌리거나(rollback), 이유를 붙여 재개(resume)한다.",
                subject=str(self._id),
            )
        if self._status is not RolloutStatus.RUNNING:
            raise IllegalStateTransition(
                f"{self._status.value} 상태에서는 진행할 수 없다.", subject=str(self._id)
            )
        if not self._results:
            raise IllegalStateTransition(
                "지금 단계의 결과를 받기 전에는 다음으로 못 간다.",
                subject=str(self._id),
            )

        latest = self._results[-1]
        blocking = [
            f for f in policy.inspect(latest) if f.severity is Severity.CRITICAL
        ]
        if blocking:
            raise IllegalStateTransition(
                "다음 단계로 갈 수 없다: "
                + "; ".join(f.describe() for f in blocking),
                subject=str(self._id),
            )

        self._current += 1
        wave = self.current_wave
        if wave is None:
            self._status = RolloutStatus.COMPLETED
            self._log(occurred_at, f"완료 — 도달 {self.coverage:.1%}")
            self._record(
                domain_events.RolloutCompleted(
                    rollout_id=self._id,
                    version=self._version,
                    succeeded=self.succeeded_count,
                    total=self.device_count,
                )
            )
            raise IllegalStateTransition(
                "마지막 단계가 끝났다. 롤아웃이 완료됐다.", subject=str(self._id)
            )

        self._log(occurred_at, f"{wave.name} 시작 ({wave.size}대)")
        self._record(
            domain_events.WaveStarted(
                rollout_id=self._id, wave=wave.name, device_count=wave.size
            )
        )
        return wave

    def complete(self, occurred_at: str) -> None:
        """남은 단계 없이 끝낸다.

        **100% 를 기다리지 않는다.** 받을 수 있는 것은 다 받았으면 끝난 것이다.
        """
        if self._status is not RolloutStatus.RUNNING:
            raise IllegalStateTransition(
                f"{self._status.value} 상태에서는 완료할 수 없다.", subject=str(self._id)
            )
        self._status = RolloutStatus.COMPLETED
        self._log(occurred_at, f"완료 — 도달 {self.coverage:.1%}")
        self._record(
            domain_events.RolloutCompleted(
                rollout_id=self._id,
                version=self._version,
                succeeded=self.succeeded_count,
                total=self.device_count,
            )
        )

    def halt(self, reason: str, occurred_at: str) -> None:
        """사람이 멈춘다."""
        if not reason.strip():
            raise InvariantViolation(
                "이유 없이 현장 배포를 멈추지 않는다.", subject="reason"
            )
        if self._status is not RolloutStatus.RUNNING:
            raise IllegalStateTransition(
                f"{self._status.value} 상태에서는 멈출 것이 없다.", subject=str(self._id)
            )
        self._status = RolloutStatus.HALTED
        self._halt_reason = reason.strip()
        self._log(occurred_at, f"중단 — {reason.strip()}")
        self._record(
            domain_events.RolloutHaltedEvent(
                rollout_id=self._id,
                wave=self.current_wave.name if self.current_wave else "",
                reason=reason.strip(),
            )
        )

    def resume(self, reason: str, occurred_at: str) -> None:
        if self._status is not RolloutStatus.HALTED:
            raise IllegalStateTransition(
                "멈춘 상태가 아니다.", subject=str(self._id)
            )
        if not reason.strip():
            raise InvariantViolation(
                "무엇을 확인했는지 없이 다시 내보내지 않는다.", subject="reason"
            )
        self._status = RolloutStatus.RUNNING
        self._halt_reason = ""
        self._log(occurred_at, f"재개 — {reason.strip()}")

    def mark_rolled_back(self, reason: str, occurred_at: str) -> None:
        """이 롤아웃이 되돌려졌다는 사실을 남긴다. (실습 6-9)

        **되돌리는 일 자체는 새 롤아웃이 한다.** 여기서는 기록만 남는다 —
        그래야 "무엇이 무엇을 되돌렸는가"가 이력에 남는다.
        """
        if not reason.strip():
            raise InvariantViolation("이유가 필요하다.", subject="reason")
        if self._status not in (RolloutStatus.RUNNING, RolloutStatus.HALTED, RolloutStatus.COMPLETED):
            raise IllegalStateTransition(
                f"{self._status.value} 상태는 되돌릴 대상이 아니다.",
                subject=str(self._id),
            )
        self._status = RolloutStatus.ROLLED_BACK
        self._log(occurred_at, f"되돌림 — {reason.strip()}")
        self._record(
            domain_events.RolloutRolledBack(
                rollout_id=self._id,
                version=self._version,
                to_version=self._previous_version,
                reason=reason.strip(),
            )
        )

    # -- 내부 --------------------------------------------------------------
    def _log(self, occurred_at: str, message: str) -> None:
        self._history.append((occurred_at, message))

    def __repr__(self) -> str:  # pragma: no cover - 디버깅 편의
        return (
            f"Rollout(id={self._id}, version={self._version}, "
            f"status={self._status.value}, coverage={self.coverage:.1%})"
        )
