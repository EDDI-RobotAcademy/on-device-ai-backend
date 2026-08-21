"""Deployment — 무엇이 어디에 올라가 있는가. (실습 5-1, 5-2, 5-8, 5-10)

이 Aggregate 는 **결정의 기록**이다. 관측은 여기 없다 (HealthWatch 가 한다).

지키는 불변식:
    - 모듈 4 의 선택 판정을 통과하지 않은 결과물은 배포 대상이 아니다
    - 격리된 배포에는 새 버전을 올릴 수 없다 — 먼저 조치해야 한다
    - **롤백 대상은 실제로 배포됐던 버전이어야 한다** — 돌아갈 곳이 없으면 롤백이 아니다
    - 롤백에도 이유가 필요하다

그리고 이 Aggregate 의 가장 중요한 설계 결정:

    **롤백은 되돌리기가 아니라 새 배포다.**

v3 에서 v1 로 돌아가면 결과는 v4 다 (내용은 v1). 버전 번호는 줄어들지 않는다.
그래야 "어제 3시에 무엇이 돌고 있었는가"에 답할 수 있다.
버전을 되감으면 그 시간의 기록이 사라진다.
"""

from __future__ import annotations

from enum import Enum

from domain.operations import events as domain_events
from domain.operations.artifact_ref import DeployedArtifactRef
from domain.operations.errors import NotDeployable, VersionNotFound
from domain.operations.identifiers import DeploymentId
from domain.operations.target import DeploymentTarget
from domain.operations.version import DeploymentVersion
from domain.shared.errors import IllegalStateTransition, InvariantViolation
from domain.shared.events import EventRecorder


class DeploymentStatus(Enum):
    DEPLOYED = "DEPLOYED"
    """돌고 있다. 설비가 이 모델의 판단을 쓰고 있다."""

    QUARANTINED = "QUARANTINED"
    """판단을 쓰지 않는다. 모델은 아직 그대로 있다 — 멈춘 것뿐이다."""

    ROLLED_BACK = "ROLLED_BACK"
    """이전 버전으로 되돌아가 다시 돌고 있다."""

    RETIRED = "RETIRED"
    """내렸다. 이 대상에서 이 모델은 끝났다."""


class Deployment(EventRecorder):
    """한 대상에 대한 배포 이력 전체."""

    __slots__ = (
        "_id",
        "_target",
        "_versions",
        "_status",
        "_quarantine_reason",
        "_history",
    )

    def __init__(self, deployment_id: DeploymentId, target: DeploymentTarget) -> None:
        super().__init__()
        self._id = deployment_id
        self._target = target
        self._versions: list[DeploymentVersion] = []
        self._status = DeploymentStatus.DEPLOYED
        self._quarantine_reason = ""
        self._history: list[tuple[str, str]] = []

    # -- 생성 --------------------------------------------------------------
    @classmethod
    def deploy(
        cls,
        deployment_id: DeploymentId,
        target: DeploymentTarget,
        artifact: DeployedArtifactRef,
        released_at: str,
        *,
        note: str = "",
        require_selected: bool = True,
    ) -> Deployment:
        """모델을 처음으로 현장에 배포한다. (실습 5-1)

        모듈 4 의 선택 판정을 통과하지 않았으면 여기서 멈춘다.
        "예산 안에 든다"를 확인하지 않은 결과물이 현장에 나가면
        그 예산은 처음부터 없었던 것과 같다.
        """
        if require_selected and not artifact.selected:
            raise NotDeployable(
                "선택 판정을 통과하지 않은 결과물은 배포 대상이 아니다: "
                + ", ".join(artifact.missing_gates),
                subject=str(deployment_id),
            )

        deployment = cls(deployment_id, target)
        version = DeploymentVersion(
            number=1, artifact=artifact, released_at=released_at, note=note
        )
        deployment._versions.append(version)
        deployment._log(released_at, f"v1 배포 — {artifact.describe()}")
        deployment._record(
            domain_events.ModelDeployed(
                deployment_id=deployment_id,
                target=target.identifier,
                version=1,
                artifact_id=artifact.artifact_id,
            )
        )
        return deployment

    # -- 조회 --------------------------------------------------------------
    @property
    def id(self) -> DeploymentId:
        return self._id

    @property
    def target(self) -> DeploymentTarget:
        return self._target

    @property
    def status(self) -> DeploymentStatus:
        return self._status

    @property
    def versions(self) -> tuple[DeploymentVersion, ...]:
        return tuple(self._versions)

    @property
    def current_version(self) -> DeploymentVersion:
        return self._versions[-1]

    @property
    def quarantine_reason(self) -> str:
        return self._quarantine_reason

    @property
    def is_serving(self) -> bool:
        """이 배포의 판단을 지금 쓰고 있는가."""
        return self._status in (
            DeploymentStatus.DEPLOYED,
            DeploymentStatus.ROLLED_BACK,
        )

    @property
    def rollback_count(self) -> int:
        return sum(1 for version in self._versions if version.is_rollback)

    @property
    def history(self) -> tuple[tuple[str, str], ...]:
        """언제 무엇을 했는가. 사건 조사는 이것부터 읽는다."""
        return tuple(self._history)

    def version_of(self, number: int) -> DeploymentVersion | None:
        return next((v for v in self._versions if v.number == number), None)

    def version_at(self, moment: str) -> DeploymentVersion | None:
        """그 시각에 돌고 있던 버전. (실습 5-2)

        **버전 번호를 되감지 않기 때문에 이 질문에 답할 수 있다.**
        """
        active = [v for v in self._versions if v.released_at <= moment]
        return active[-1] if active else None

    def render(self) -> str:
        lines = [
            f"배포 {self._id}  {self._target.describe()}  → {self._status.value}",
            "-" * 74,
        ]
        lines += [f"  {version.describe()}" for version in self._versions]
        if self._quarantine_reason:
            lines.append("")
            lines.append(f"  격리 사유: {self._quarantine_reason}")
        return "\n".join(lines)

    # -- 행위 --------------------------------------------------------------
    def release(
        self,
        artifact: DeployedArtifactRef,
        released_at: str,
        *,
        note: str = "",
        require_selected: bool = True,
    ) -> DeploymentVersion:
        """새 버전을 올린다. (실습 5-2)"""
        self._guard_serving("새 버전 배포")
        if require_selected and not artifact.selected:
            raise NotDeployable(
                "선택 판정을 통과하지 않은 결과물은 배포 대상이 아니다: "
                + ", ".join(artifact.missing_gates),
                subject=str(self._id),
            )

        version = DeploymentVersion(
            number=self.current_version.number + 1,
            artifact=artifact,
            released_at=released_at,
            note=note,
        )
        self._versions.append(version)
        self._status = DeploymentStatus.DEPLOYED
        self._log(released_at, f"{version.label} 배포 — {artifact.describe()}")
        self._record(
            domain_events.VersionReleased(
                deployment_id=self._id,
                version=version.number,
                artifact_id=artifact.artifact_id,
                previous_version=version.number - 1,
            )
        )
        return version

    def quarantine(self, reason: str, occurred_at: str) -> None:
        """판단을 멈춘다. (실습 5-8)

        모델을 내리지 않는다. 되돌리지도 않는다. **쓰지 않을 뿐이다.**
        이전 모델이 더 나으리라는 보장이 없기 때문이다 —
        입력이 변한 것이라면 이전 모델도 똑같이 틀린다.
        """
        if self._status is DeploymentStatus.RETIRED:
            raise IllegalStateTransition(
                "내려간 배포는 격리 대상이 아니다.", subject=str(self._id)
            )
        if not reason.strip():
            raise InvariantViolation(
                "이유 없이 현장 판단을 멈추지 않는다.", subject="reason"
            )
        if self._status is DeploymentStatus.QUARANTINED:
            return  # 이미 멈춰 있다. 두 번 멈추지 않는다.

        self._status = DeploymentStatus.QUARANTINED
        self._quarantine_reason = reason.strip()
        self._log(occurred_at, f"격리 — {reason.strip()}")
        self._record(
            domain_events.DeploymentQuarantined(
                deployment_id=self._id,
                version=self.current_version.number,
                reason=reason.strip(),
            )
        )

    def rollback(self, to_version: int, reason: str, occurred_at: str) -> DeploymentVersion:
        """이전 버전으로 되돌린다. (실습 5-10)

        **되돌리기가 아니라 새 배포다.** 버전 번호는 계속 올라간다.
        """
        if self._status is DeploymentStatus.RETIRED:
            raise IllegalStateTransition(
                "내려간 배포는 롤백 대상이 아니다.", subject=str(self._id)
            )
        if not reason.strip():
            raise InvariantViolation(
                "이유 없는 롤백은 다음 사람에게 아무것도 알려주지 않는다.",
                subject="reason",
            )

        source = self.version_of(to_version)
        if source is None:
            raise VersionNotFound(
                f"v{to_version} 은 이 대상에 배포된 적이 없다. "
                "돌아갈 곳이 없으면 롤백이 아니다.",
                subject=f"v{to_version}",
            )
        if to_version == self.current_version.number:
            raise IllegalStateTransition(
                f"이미 v{to_version} 이 돌고 있다.", subject=str(self._id)
            )

        from_version = self.current_version.number
        version = DeploymentVersion(
            number=from_version + 1,
            artifact=source.artifact,
            released_at=occurred_at,
            note=reason.strip(),
            rolled_back_from=from_version,
        )
        self._versions.append(version)
        self._status = DeploymentStatus.ROLLED_BACK
        self._quarantine_reason = ""
        self._log(
            occurred_at,
            f"{version.label} 롤백 (v{from_version} → v{to_version} 의 내용) — {reason.strip()}",
        )
        self._record(
            domain_events.DeploymentRolledBack(
                deployment_id=self._id,
                from_version=from_version,
                to_version=to_version,
                new_version=version.number,
                reason=reason.strip(),
            )
        )
        return version

    def resume(self, reason: str, occurred_at: str) -> None:
        """격리를 푼다. 원인을 확인했을 때만."""
        if self._status is not DeploymentStatus.QUARANTINED:
            raise IllegalStateTransition(
                "격리 상태가 아니다.", subject=str(self._id)
            )
        if not reason.strip():
            raise InvariantViolation(
                "무엇을 확인했는지 없이 판단을 다시 켜지 않는다.", subject="reason"
            )
        self._status = DeploymentStatus.DEPLOYED
        self._quarantine_reason = ""
        self._log(occurred_at, f"격리 해제 — {reason.strip()}")
        self._record(
            domain_events.DeploymentResumed(
                deployment_id=self._id,
                version=self.current_version.number,
                reason=reason.strip(),
            )
        )

    def retire(self, reason: str, occurred_at: str) -> None:
        if not reason.strip():
            raise InvariantViolation("이유가 필요하다.", subject="reason")
        self._status = DeploymentStatus.RETIRED
        self._log(occurred_at, f"내림 — {reason.strip()}")

    # -- 내부 --------------------------------------------------------------
    def _guard_serving(self, action: str) -> None:
        if self._status is DeploymentStatus.QUARANTINED:
            raise IllegalStateTransition(
                f"격리 상태에서는 '{action}' 을 할 수 없다. "
                "원인을 확인해 resume 하거나, rollback 으로 되돌린 뒤에 한다.",
                subject=str(self._id),
            )
        if self._status is DeploymentStatus.RETIRED:
            raise IllegalStateTransition(
                f"내려간 배포에는 '{action}' 을 할 수 없다.", subject=str(self._id)
            )

    def _log(self, occurred_at: str, message: str) -> None:
        self._history.append((occurred_at, message))

    def __repr__(self) -> str:  # pragma: no cover - 디버깅 편의
        return (
            f"Deployment(id={self._id}, status={self._status.value}, "
            f"version={self.current_version.label})"
        )
