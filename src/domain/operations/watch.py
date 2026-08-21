"""HealthWatch — 배포 하나를 지켜본 기록. (실습 5-3 ~ 5-7, 5-9, 5-11)

Deployment 와 나눈 이유는 하나다.

    **관측 기록은 배포보다 오래 산다.**

배포가 롤백되고 격리되고 내려가도, "왜 그랬는가"는 남아야 한다.
그 기록이 다음 모델의 학습 데이터를 정하고, 다음 예산을 정한다.

이 Aggregate 는 Deployment 를 **ID 로만** 참조한다.
두 Aggregate 가 서로의 내부를 만지기 시작하면 하나로 합쳐야 한다 — 그건 다른 설계다.
"""

from __future__ import annotations

from domain.operations import events as domain_events
from domain.operations.errors import NoObservationRecorded
from domain.operations.health import HealthReport, HealthTimeline
from domain.operations.identifiers import DeploymentId, IncidentId, WatchId
from domain.operations.incident import Incident, IncidentPolicy
from domain.shared.errors import InvariantViolation
from domain.shared.events import EventRecorder
from domain.shared.inspection import Verdict


class HealthWatch(EventRecorder):
    """한 배포에 대한 관측 기록."""

    __slots__ = (
        "_id",
        "_deployment_id",
        "_baseline_p95_ms",
        "_baseline_mix",
        "_reports",
        "_incidents",
    )

    def __init__(
        self,
        watch_id: WatchId,
        deployment_id: DeploymentId,
        *,
        baseline_p95_ms: float,
        baseline_mix: dict[str, float],
    ) -> None:
        super().__init__()
        self._id = watch_id
        self._deployment_id = deployment_id
        self._baseline_p95_ms = baseline_p95_ms
        self._baseline_mix = dict(baseline_mix)
        self._reports: list[HealthReport] = []
        self._incidents: list[Incident] = []

    # -- 생성 --------------------------------------------------------------
    @classmethod
    def open(
        cls,
        watch_id: WatchId,
        deployment_id: DeploymentId,
        *,
        baseline_p95_ms: float,
        baseline_mix: dict[str, float],
    ) -> HealthWatch:
        """배포와 **동시에** 시작한다.

        나중에 시작하면 그 사이 구간은 영영 비어 있다.
        "언제부터 이상해졌죠?"에 "관측을 켠 뒤부터는요"라고 답하게 된다.
        """
        if baseline_p95_ms < 0:
            raise InvariantViolation(
                "기준 지연시간은 음수일 수 없다.", subject="baseline_p95_ms"
            )
        watch = cls(
            watch_id,
            deployment_id,
            baseline_p95_ms=baseline_p95_ms,
            baseline_mix=baseline_mix,
        )
        watch._record(
            domain_events.HealthWatchOpened(
                watch_id=watch_id, deployment_id=deployment_id
            )
        )
        return watch

    # -- 조회 --------------------------------------------------------------
    @property
    def id(self) -> WatchId:
        return self._id

    @property
    def deployment_id(self) -> DeploymentId:
        return self._deployment_id

    @property
    def baseline_p95_ms(self) -> float:
        return self._baseline_p95_ms

    @property
    def baseline_mix(self) -> dict[str, float]:
        return dict(self._baseline_mix)

    @property
    def timeline(self) -> HealthTimeline:
        return HealthTimeline(reports=tuple(self._reports))

    @property
    def reports(self) -> tuple[HealthReport, ...]:
        return tuple(self._reports)

    @property
    def latest(self) -> HealthReport:
        if not self._reports:
            raise NoObservationRecorded(
                "관측이 하나도 없다. 배포는 됐는데 아무도 안 보고 있다.",
                subject=str(self._id),
            )
        return self._reports[-1]

    @property
    def incidents(self) -> tuple[Incident, ...]:
        return tuple(self._incidents)

    @property
    def open_incidents(self) -> tuple[Incident, ...]:
        return tuple(incident for incident in self._incidents if incident.is_open)

    @property
    def has_open_incident(self) -> bool:
        return bool(self.open_incidents)

    def incident_of(self, incident_id: IncidentId) -> Incident | None:
        return next(
            (i for i in self._incidents if i.incident_id == incident_id), None
        )

    def render(self) -> str:
        lines = [
            f"관측 {self._id}  (배포 {self._deployment_id})  창 {len(self._reports)}개",
            f"  기준 p95 {self._baseline_p95_ms:.4f}ms  "
            f"기준 분포 "
            + "  ".join(f"{k} {v:.1%}" for k, v in sorted(self._baseline_mix.items())),
            "",
            self.timeline.render(),
        ]
        if self._incidents:
            lines.append("")
            lines.append("사건:")
            lines += [f"  {incident.render()}" for incident in self._incidents]
        return "\n".join(lines)

    # -- 행위 --------------------------------------------------------------
    def record(self, report: HealthReport) -> None:
        """창 하나의 관측 결과를 남긴다. (실습 5-3, 5-4)

        창은 시간 순으로만 들어온다. 거꾸로 들어오면 시간선이 거짓말을 한다.
        """
        if self._reports:
            last = self._reports[-1]
            if report.window.started_at < last.window.started_at:
                raise InvariantViolation(
                    f"'{report.window.label}' 이 직전 창보다 앞선다. "
                    "시간선이 뒤엉키면 '언제부터'에 답할 수 없다.",
                    subject=report.window.label,
                )
            if report.window.label == last.window.label:
                self._reports[-1] = report  # 같은 창을 다시 재면 덮어쓴다
                return

        self._reports.append(report)
        self._record(
            domain_events.ObservationRecorded(
                watch_id=self._id,
                window_label=report.window.label,
                verdict=report.verdict,
                sample_count=report.window.sample_count,
            )
        )

    def open_incident(
        self, incident_id: IncidentId, report: HealthReport, policy: IncidentPolicy
    ) -> Incident | None:
        """관측 결과가 사건이면 사건으로 등록한다. (실습 5-8)

        사건이 아니면 None 을 돌려준다 — 아무 일도 없었던 창까지 사건으로 만들지 않는다.
        """
        if report.verdict is Verdict.PASSED:
            return None

        incident = Incident(
            incident_id=incident_id,
            kind=policy.kind_of(report),
            opened_at=report.window.started_at,
            window_label=report.window.label,
            deployment_version=report.deployment_version,
            findings=report.findings,
        )
        self._incidents.append(incident)
        self._record(
            domain_events.IncidentOpened(
                watch_id=self._id,
                incident_id=incident_id,
                kind=incident.kind.value,
                window_label=report.window.label,
            )
        )
        return incident

    def rebaseline(self, mix: dict[str, float], reason: str) -> dict[str, float]:
        """기준 예측 분포를 현장 안정 구간으로 다시 잡는다. (실습 5-6)

        처음 기준은 모듈 4 의 **평가 데이터**에서 온다.
        그런데 평가 데이터와 현장은 애초에 구성이 다르다 —
        평가 집합에는 사건이 골고루 들어 있지만(실습 3-8), 현장은 대개 평온하다.

            평가 때   FAULT 4.5%
            현장 1일차 FAULT 0.8%

        이 상태로 두면 현장 분포는 **처음부터** 기준과 어긋나 있고,
        진짜 변화가 와도 그 어긋남에 묻힌다.

        그래서 배포 직후 **아무 일도 없었던 구간**을 새 기준으로 못박는다.
        단, 그 구간이 정말 평온했는지는 사람이 확인해야 한다 —
        이상한 구간을 기준으로 잡으면 그 이상이 정상이 된다.
        """
        if not mix:
            raise InvariantViolation(
                "빈 분포를 기준으로 삼을 수 없다.", subject="mix"
            )
        if not reason.strip():
            raise InvariantViolation(
                "기준을 바꾸는 것은 판정 기준을 바꾸는 것이다. 이유를 남긴다.",
                subject="reason",
            )
        previous = dict(self._baseline_mix)
        self._baseline_mix = dict(mix)
        self._record(
            domain_events.BaselineReanchored(
                watch_id=self._id,
                reason=reason.strip(),
                previous=tuple(sorted(previous.items())),
                current=tuple(sorted(self._baseline_mix.items())),
            )
        )
        return previous

    def request_retraining(self, urgency: str, reasons: tuple[str, ...]) -> None:
        """재학습이 필요하다는 신호를 남긴다. (실습 5-11)

        판정 자체는 RetrainingPolicy 가 한다. Aggregate 는 **그 사실을 알린다.**
        이 사건을 받는 쪽이 모듈 1 의 데이터 수집이다 — 순환은 여기서 닫힌다.
        """
        if not reasons:
            raise InvariantViolation(
                "근거 없는 재학습 요청은 남기지 않는다.", subject="reasons"
            )
        self._record(
            domain_events.RetrainingRequested(
                watch_id=self._id,
                deployment_id=self._deployment_id,
                urgency=urgency,
                reasons=reasons,
            )
        )

    def resolve_incident(self, incident_id: IncidentId, resolution: str) -> Incident:
        for index, incident in enumerate(self._incidents):
            if incident.incident_id == incident_id:
                resolved = incident.resolved(resolution)
                self._incidents[index] = resolved
                self._record(
                    domain_events.IncidentResolved(
                        watch_id=self._id,
                        incident_id=incident_id,
                        resolution=resolved.resolution,
                    )
                )
                return resolved
        raise InvariantViolation(
            f"사건 '{incident_id}' 이 이 관측에 없다.", subject=str(incident_id)
        )

    def __repr__(self) -> str:  # pragma: no cover - 디버깅 편의
        return (
            f"HealthWatch(id={self._id}, windows={len(self._reports)}, "
            f"incidents={len(self._incidents)})"
        )
