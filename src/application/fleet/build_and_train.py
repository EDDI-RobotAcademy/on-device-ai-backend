"""Cloud에서 다시 학습 데이터를 만들고, 클라우드에서 학습시킨다. (실습 6-4, 6-5)

여기가 순환의 위쪽 반이다.

    현장 데이터 → 학습 데이터셋 → 원격 학습 → 결과물
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from application.data.dto import FindingView
from application.fleet.dto import DatasetBuildView, TrainingJobView
from application.fleet.support import load_fleet
from application.shared.errors import ConflictingRequest
from application.shared.ports import EventPublisher
from domain.fleet.dataset_build import (
    DatasetBuildPolicy,
    DatasetBuildSpec,
    SourceWindow,
)
from domain.fleet.object_key import ObjectKey
from domain.fleet.ports import (
    DeviceRegistry,
    FleetRepository,
    ObjectStore,
    RemoteTrainingGateway,
)
from domain.fleet.training_job import (
    ComputeSpec,
    RemoteTrainingJob,
    TrainingBudgetPolicy,
)


@dataclass(frozen=True, slots=True)
class BuildDatasetCommand:
    """현장 데이터로 학습 데이터셋을 만든다. (실습 6-4)"""

    fleet_id: str
    build_id: str
    window: SourceWindow
    record_counts: dict[str, int]
    labeled_counts: dict[str, int]
    label_distribution: dict[str, int]
    policy: DatasetBuildPolicy = field(default_factory=DatasetBuildPolicy)
    include_devices: tuple[str, ...] = ()
    """비우면 **학습에 써도 되는 디바이스 전부**를 쓴다."""


class BuildTrainingDataset:
    """어느 디바이스의 어느 구간을 쓸 것인가.

    **격리된 디바이스는 자동으로 빠진다.** 그리고 뺐다는 사실이 기록에 남는다.
    """

    def __init__(
        self,
        fleets: FleetRepository,
        store: ObjectStore,
        registry: DeviceRegistry,
    ) -> None:
        self._fleets = fleets
        self._store = store
        self._registry = registry

    def execute(self, command: BuildDatasetCommand) -> DatasetBuildView:
        fleet = load_fleet(self._fleets, command.fleet_id)

        candidates = command.include_devices or tuple(
            d.device_id for d in fleet.devices
        )
        included: list[str] = []
        excluded: list[tuple[str, str]] = []
        for device_id in candidates:
            device = fleet.device(device_id)
            if device.is_trainable_source:
                included.append(device_id)
            else:
                excluded.append(
                    (
                        device_id,
                        f"{device.status.value} — 이상한 상태에서 낸 판단은 "
                        "학습에 넣지 않는다 (실습 5-8)",
                    )
                )

        if not included:
            raise ConflictingRequest(
                "학습에 쓸 수 있는 디바이스가 하나도 없다.", subject=command.fleet_id
            )

        spec = DatasetBuildSpec(
            build_id=command.build_id,
            window=command.window,
            device_ids=tuple(included),
            excluded_devices=tuple(excluded),
            record_counts={
                d: n for d, n in command.record_counts.items() if d in included
            },
            labeled_counts={
                d: n for d, n in command.labeled_counts.items() if d in included
            },
            label_distribution=command.label_distribution,
        )
        check = command.policy.inspect(spec)

        uri = ""
        if check.can_build:
            key = ObjectKey(
                prefix="datasets",
                partitions=(("build", command.build_id),),
                filename="manifest.json",
            )
            uri = self._store.put(key, _manifest(spec))

        return DatasetBuildView.of(str(fleet.id), check, uri)


@dataclass(frozen=True, slots=True)
class SubmitTrainingCommand:
    """SageMaker에서 새로운 모델을 학습시켜라. (실습 6-5)"""

    job_id: str
    dataset_uri: str
    output_uri: str
    compute: ComputeSpec
    hyperparameters: dict[str, str] = field(default_factory=dict)
    policy: TrainingBudgetPolicy = field(default_factory=TrainingBudgetPolicy)


class SubmitTrainingJob:
    """제출하고 **즉시 돌아온다.** (CLAUDE.md §11)"""

    def __init__(
        self,
        gateway: RemoteTrainingGateway,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._gateway = gateway
        self._publisher = publisher

    def execute(self, command: SubmitTrainingCommand) -> TrainingJobView:
        planned = RemoteTrainingJob(
            job_id=command.job_id,
            dataset_uri=command.dataset_uri,
            output_uri=command.output_uri,
            compute=command.compute,
        )
        findings = command.policy.inspect_submission(planned)
        from domain.shared.inspection import Severity

        if any(f.severity is Severity.CRITICAL for f in findings):
            # 예산을 넘는 학습은 **제출하지 않는다.** 제출한 뒤 멈추면 이미 과금됐다.
            return TrainingJobView.of(
                planned, tuple(FindingView.of(f) for f in findings)
            )

        job = self._gateway.submit(
            command.job_id,
            dataset_uri=command.dataset_uri,
            output_uri=command.output_uri,
            compute=command.compute,
            hyperparameters=command.hyperparameters,
        )
        if self._publisher:
            from domain.fleet import events as domain_events

            self._publisher.publish(
                [
                    domain_events.TrainingJobSubmitted(
                        job_id=job.job_id,
                        dataset_uri=job.dataset_uri,
                        instance_type=job.compute.instance_type,
                    )
                ]
            )
        return TrainingJobView.of(job, tuple(FindingView.of(f) for f in findings))


@dataclass(frozen=True, slots=True)
class PollTrainingQuery:
    job_id: str
    policy: TrainingBudgetPolicy = field(default_factory=TrainingBudgetPolicy)


class PollTrainingJob:
    """지금 어떻게 됐는지 물어본다.

    **기다리지 않는다.** 상태를 돌려주고, 끝났으면 결과까지 판정한다.
    """

    def __init__(
        self,
        gateway: RemoteTrainingGateway,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._gateway = gateway
        self._publisher = publisher

    def execute(self, query: PollTrainingQuery) -> TrainingJobView:
        job = self._gateway.describe(query.job_id)
        findings = query.policy.inspect_result(job)

        if job.is_terminal and self._publisher:
            from domain.fleet import events as domain_events

            self._publisher.publish(
                [
                    domain_events.TrainingJobFinished(
                        job_id=job.job_id,
                        status=job.status.value,
                        artifact_uri=job.artifact_uri,
                    )
                ]
            )
        return TrainingJobView.of(job, tuple(FindingView.of(f) for f in findings))


@dataclass(frozen=True, slots=True)
class StopTrainingCommand:
    job_id: str
    reason: str


class StopTrainingJob:
    def __init__(self, gateway: RemoteTrainingGateway) -> None:
        self._gateway = gateway

    def execute(self, command: StopTrainingCommand) -> TrainingJobView:
        if not command.reason.strip():
            raise ConflictingRequest(
                "이유 없이 학습을 멈추지 않는다. 이미 쓴 비용이 있다.",
                subject=command.job_id,
            )
        return TrainingJobView.of(self._gateway.stop(command.job_id, command.reason))


def _manifest(spec: DatasetBuildSpec) -> bytes:
    """데이터셋 명세를 그대로 남긴다.

    **무엇을 넣었고 무엇을 뺐는지**가 6개월 뒤 계보의 한 칸이 된다 (실습 6-10).
    """
    return json.dumps(
        {
            "build_id": spec.build_id,
            "window": {
                "started_at": spec.window.started_at,
                "ended_at": spec.window.ended_at,
                "reason": spec.window.reason,
            },
            "devices": list(spec.device_ids),
            "excluded": [
                {"device_id": d, "reason": r} for d, r in spec.excluded_devices
            ],
            "record_counts": dict(spec.record_counts),
            "labeled_counts": dict(spec.labeled_counts),
            "label_distribution": dict(spec.label_distribution),
        },
        ensure_ascii=False,
        indent=2,
    ).encode()
