"""모델을 처음으로 현장에 배포하라. (실습 5-1, 5-2)

배포는 파일을 복사하는 일이 아니다. 네 가지가 함께 나가야 한다.

    결과물     모듈 4 가 고른 것
    전처리     모듈 3 이 뽑은 정규화 통계
    기준       모듈 4 가 잰 지연시간과 예측 분포
    관측       배포와 **동시에** 시작하는 HealthWatch

마지막 것이 핵심이다. 관측을 나중에 켜면 그 사이 구간은 영영 비어 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.operations.artifact_mapper import artifact_from
from application.operations.dto import DeploymentView, ReleaseCheckView
from application.operations.support import commit, load_deployment, moment
from application.optimization.support import load_run as load_optimization_run
from application.shared.ports import Clock, EventPublisher
from domain.model.ports import TrainingRunRepository
from domain.operations.deployment import Deployment
from domain.operations.identifiers import DeploymentId, WatchId
from domain.operations.ports import (
    DeploymentRepository,
    HealthWatchRepository,
)
from domain.operations.release_check import ReleasePolicy
from domain.operations.target import DeploymentTarget
from domain.operations.watch import HealthWatch
from domain.optimization.ports import OptimizationRunRepository


@dataclass(frozen=True, slots=True)
class DeployModelCommand:
    deployment_id: str
    optimization_run_id: str
    target: DeploymentTarget
    training_run_id: str | None = None
    """전처리 통계를 여기서 가져온다. 없으면 배포 점검이 막는다."""

    artifact_label: str | None = None
    watch_id: str | None = None
    note: str = ""
    released_at: str | None = None
    """기록할 시각. 비우면 지금이다. 지난 로그를 다시 넣을 때 못박는다."""

    require_selected: bool = True
    policy: ReleasePolicy = field(default_factory=ReleasePolicy)


@dataclass(frozen=True, slots=True)
class DeployModelResult:
    deployment: DeploymentView
    check: ReleaseCheckView
    watch_id: str


class DeployModel:
    def __init__(
        self,
        deployments: DeploymentRepository,
        watches: HealthWatchRepository,
        optimization_runs: OptimizationRunRepository,
        training_runs: TrainingRunRepository,
        clock: Clock,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._deployments = deployments
        self._watches = watches
        self._optimization_runs = optimization_runs
        self._training_runs = training_runs
        self._clock = clock
        self._publisher = publisher

    def execute(self, command: DeployModelCommand) -> DeployModelResult:
        artifact = self._artifact(command)
        check = command.policy.inspect(artifact, command.target, previous_versions=0)

        deployment_id = DeploymentId.of(command.deployment_id)
        deployment = Deployment.deploy(
            deployment_id,
            command.target,
            artifact,
            moment(self._clock, command.released_at),
            note=command.note,
            require_selected=command.require_selected,
        )
        commit(self._deployments, deployment, self._publisher)

        # 관측은 배포와 **동시에** 시작한다.
        watch_id = WatchId.of(command.watch_id or f"watch-{command.deployment_id}")
        watch = HealthWatch.open(
            watch_id,
            deployment_id,
            baseline_p95_ms=artifact.expected_p95_ms,
            baseline_mix=dict(artifact.expected_class_mix),
        )
        commit(self._watches, watch, self._publisher)

        return DeployModelResult(
            deployment=DeploymentView.of(deployment),
            check=ReleaseCheckView.of(str(deployment.id), check),
            watch_id=str(watch_id),
        )

    def _artifact(self, command: DeployModelCommand):  # noqa: ANN202
        run = load_optimization_run(
            self._optimization_runs, command.optimization_run_id
        )
        training_run = None
        if command.training_run_id:
            from application.model.support import load_run as load_training_run

            training_run = load_training_run(
                self._training_runs, command.training_run_id
            )
        return artifact_from(run, training_run, label=command.artifact_label)


@dataclass(frozen=True, slots=True)
class ReleaseVersionCommand:
    """새 버전을 올린다. (실습 5-2)"""

    deployment_id: str
    optimization_run_id: str
    training_run_id: str | None = None
    artifact_label: str | None = None
    note: str = ""
    released_at: str | None = None
    require_selected: bool = True
    policy: ReleasePolicy = field(default_factory=ReleasePolicy)


class ReleaseVersion:
    def __init__(
        self,
        deployments: DeploymentRepository,
        optimization_runs: OptimizationRunRepository,
        training_runs: TrainingRunRepository,
        clock: Clock,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._deployments = deployments
        self._optimization_runs = optimization_runs
        self._training_runs = training_runs
        self._clock = clock
        self._publisher = publisher

    def execute(self, command: ReleaseVersionCommand) -> DeployModelResult:
        deployment = load_deployment(self._deployments, command.deployment_id)

        run = load_optimization_run(
            self._optimization_runs, command.optimization_run_id
        )
        training_run = None
        if command.training_run_id:
            from application.model.support import load_run as load_training_run

            training_run = load_training_run(
                self._training_runs, command.training_run_id
            )
        artifact = artifact_from(run, training_run, label=command.artifact_label)

        check = command.policy.inspect(
            artifact,
            deployment.target,
            previous_versions=len(deployment.versions),
        )
        deployment.release(
            artifact,
            moment(self._clock, command.released_at),
            note=command.note,
            require_selected=command.require_selected,
        )
        commit(self._deployments, deployment, self._publisher)

        return DeployModelResult(
            deployment=DeploymentView.of(deployment),
            check=ReleaseCheckView.of(str(deployment.id), check),
            watch_id="",
        )


@dataclass(frozen=True, slots=True)
class GetDeploymentQuery:
    deployment_id: str


class GetDeployment:
    def __init__(self, deployments: DeploymentRepository) -> None:
        self._deployments = deployments

    def execute(self, query: GetDeploymentQuery) -> DeploymentView:
        return DeploymentView.of(load_deployment(self._deployments, query.deployment_id))


class ListDeployments:
    def __init__(self, deployments: DeploymentRepository) -> None:
        self._deployments = deployments

    def execute(self) -> tuple[DeploymentView, ...]:
        return tuple(DeploymentView.of(d) for d in self._deployments.list_all())
