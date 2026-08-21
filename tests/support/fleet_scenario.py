"""모듈 6 실습 시나리오 빌더.

모듈 6 은 앞의 다섯 모듈 **전부** 위에 선다.
그리고 여기서 처음으로 **AWS 가 실제로 호출된다** — moto 안에서.

    moto 는 boto3 요청을 가로채 메모리에서 처리한다.
    가짜 클라이언트를 만들어 "호출됐다"만 확인하는 것과 다르다 —
    **API 이름이 틀리면 여기서 터진다.**
"""

from __future__ import annotations

from types import SimpleNamespace

from application.fleet.build_and_train import (
    BuildDatasetCommand,
    PollTrainingQuery,
    SubmitTrainingCommand,
)
from application.fleet.ingest_uplink import (
    CreateFleetCommand,
    IngestUplinkCommand,
    InspectLakeCommand,
    MarkDeviceCommand,
    SummarizeFleetQuery,
    SweepStaleCommand,
)
from application.fleet.release_and_rollout import (
    AdvanceRolloutCommand,
    ApplyOutcomesCommand,
    CollectWaveCommand,
    PlanRolloutCommand,
    PromoteReleaseCommand,
    PublishReleaseCommand,
    RollbackRolloutCommand,
    TraceLineageQuery,
)
from domain.fleet.dataset_build import SourceWindow
from domain.fleet.device import Device, DeviceStatus
from domain.fleet.release import ReleaseBundle, ReleaseChannel
from domain.fleet.training_job import ComputeSpec
from domain.fleet.uplink import UplinkBatch, UplinkKind
from infrastructure.aws.config import AwsConfig
from infrastructure.config.container import FleetContainer
from infrastructure.edge.ota_simulator import (
    FleetResponseProfile,
    SimulatedFleetOtaGateway,
)

FLEET_ID = "line3"
DEVICE_COUNT = 24
GROUPS = ("pilot", "line-a", "line-b")


def response_profile(**overrides) -> FleetResponseProfile:  # noqa: ANN003
    """기본 현장 — 대부분 성공하고, 8% 는 꺼져 있고, 5% 는 아직 답이 없다."""
    base: dict[str, object] = dict(
        failure_rate=0.0, offline_rate=0.08, pending_rate=0.05
    )
    base.update(overrides)
    return FleetResponseProfile(**base)  # type: ignore[arg-type]


def aws_config() -> AwsConfig:
    return AwsConfig(
        region="ap-northeast-2",
        lake_bucket="ondevice-ai-lake",
        artifact_bucket="ondevice-ai-artifacts",
        device_table="ondevice-ai-devices",
        uplink_table="ondevice-ai-uplinks",
    )


def devices(count: int = DEVICE_COUNT, *, version: str = "v1.0.0") -> tuple[Device, ...]:
    """pilot 2대, line-a 8대, 나머지 line-b.

    **단계적 배포를 하려면 그룹이 있어야 한다.** 그룹 없이 나누면 무작위가 된다.
    """
    result = []
    for index in range(count):
        group = GROUPS[0] if index < 2 else GROUPS[1] if index < 10 else GROUPS[2]
        result.append(
            Device(
                device_id=f"DEV-{index:02d}",
                group=group,
                current_version=version,
                last_seen_at="2026-05-23 09:00:00",
                site="LINE-3",
            )
        )
    return tuple(result)


def bundle(version: str = "v2.0.0", **overrides) -> ReleaseBundle:  # noqa: ANN003
    base: dict[str, object] = dict(
        release_id=f"rel-{version}",
        version=version,
        model_version_id="mv-power-opt-cnn1d-s42",
        artifact_uri=f"s3://ondevice-ai-artifacts/{version}/model.tflite",
        artifact_bytes=11_724,
        checksum="deadbeef" * 8,
        runtime="TFLITE",
        precision="FP16",
        class_labels=("FAULT", "OVERLOAD", "NORMAL"),
        input_fields=(
            "active_power_kw",
            "reactive_power_kvar",
            "current_a",
            "voltage_v",
            "temperature_c",
            "spindle_rpm",
        ),
        normalization={"active_power_kw": (147.8, 39.8)},
        expected_p95_ms=0.0031,
        expected_class_mix={"NORMAL": 0.78, "OVERLOAD": 0.21, "FAULT": 0.01},
        sample_interval_seconds=10,
        window_length=30,
        channel=ReleaseChannel.CANARY,
        built_at="2026-05-24 10:00:00",
        source_build_id="build-2026-05-24",
        source_job_id="train-2026-05-24",
    )
    base.update(overrides)
    return ReleaseBundle(**base)  # type: ignore[arg-type]


def batch(device_id: str = "DEV-02", **overrides) -> UplinkBatch:  # noqa: ANN003
    base: dict[str, object] = dict(
        device_id=device_id,
        kind=UplinkKind.INFERENCE_LOG,
        window_start="2026-05-23 09:00:00",
        window_end="2026-05-23 09:59:59",
        record_count=360,
        payload_bytes=72_000,
        checksum="c0ffee",
        fields=("occurred_at", "predicted_label", "confidence", "latency_ms"),
    )
    base.update(overrides)
    return UplinkBatch(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Use Case 호출 거들기
# ---------------------------------------------------------------------------
def create(fleet: FleetContainer, **overrides):  # noqa: ANN003, ANN201
    body: dict[str, object] = dict(
        fleet_id=FLEET_ID, name="3라인 전력 감시", devices=devices()
    )
    body.update(overrides)
    return fleet.create_fleet().execute(CreateFleetCommand(**body))  # type: ignore[arg-type]


def ingest(fleet: FleetContainer, uplink: UplinkBatch, *, part: int = 0, **overrides):  # noqa: ANN003, ANN201
    body: dict[str, object] = dict(
        fleet_id=FLEET_ID,
        batch=uplink,
        body=b'{"record":1}' * 600,
        part=part,
    )
    body.update(overrides)
    return fleet.ingest_uplink().execute(IngestUplinkCommand(**body))  # type: ignore[arg-type]


def fill_lake(fleet: FleetContainer, *, devices_count: int = 4, hours: int = 3) -> int:
    """여러 디바이스가 여러 시간대에 올린 상태를 만든다."""
    written = 0
    for index in range(devices_count):
        device_id = f"DEV-{index:02d}"
        for hour in range(hours):
            uplink = batch(
                device_id=device_id,
                window_start=f"2026-05-23 {hour + 9:02d}:00:00",
                window_end=f"2026-05-23 {hour + 9:02d}:59:59",
            )
            if ingest(fleet, uplink, part=hour).accepted:
                written += 1
    return written


def inspect_lake(fleet: FleetContainer, **overrides):  # noqa: ANN003, ANN201
    body: dict[str, object] = dict(fleet_id=FLEET_ID)
    body.update(overrides)
    return fleet.inspect_lake_layout().execute(InspectLakeCommand(**body))  # type: ignore[arg-type]


def summarize(fleet: FleetContainer, **overrides):  # noqa: ANN003, ANN201
    body: dict[str, object] = dict(fleet_id=FLEET_ID)
    body.update(overrides)
    return fleet.summarize_fleet().execute(SummarizeFleetQuery(**body))  # type: ignore[arg-type]


def mark(fleet: FleetContainer, device_id: str, status: DeviceStatus, note: str = ""):  # noqa: ANN201
    return fleet.mark_device().execute(
        MarkDeviceCommand(
            fleet_id=FLEET_ID, device_id=device_id, status=status, note=note
        )
    )


def sweep(fleet: FleetContainer, **overrides):  # noqa: ANN003, ANN201
    body: dict[str, object] = dict(
        fleet_id=FLEET_ID,
        now="2026-05-25 09:00:00",
        stale_after="2026-05-24 09:00:00",
        unreachable_after="2026-05-23 09:00:00",
    )
    body.update(overrides)
    return fleet.sweep_stale_devices().execute(SweepStaleCommand(**body))  # type: ignore[arg-type]


def build_dataset(fleet: FleetContainer, **overrides):  # noqa: ANN003, ANN201
    body: dict[str, object] = dict(
        fleet_id=FLEET_ID,
        build_id="build-2026-05-24",
        window=SourceWindow(
            started_at="2026-05-22 00:00:00",
            ended_at="2026-05-23 23:59:59",
            reason="드리프트 시작 이후 구간 (실습 5-7)",
        ),
        record_counts={f"DEV-{n:02d}": 900 for n in range(DEVICE_COUNT)},
        labeled_counts={f"DEV-{n:02d}": 110 for n in range(DEVICE_COUNT)},
        label_distribution={"NORMAL": 1_900, "OVERLOAD": 600, "FAULT": 140},
    )
    body.update(overrides)
    return fleet.build_training_dataset().execute(BuildDatasetCommand(**body))  # type: ignore[arg-type]


def submit_training(fleet: FleetContainer, **overrides):  # noqa: ANN003, ANN201
    body: dict[str, object] = dict(
        job_id="train-2026-05-24",
        dataset_uri="s3://ondevice-ai-lake/datasets/build=build-2026-05-24/",
        output_uri="s3://ondevice-ai-artifacts/train-2026-05-24/",
        compute=ComputeSpec(
            instance_type="ml.m5.large",
            instance_count=1,
            max_runtime_seconds=3_600,
            hourly_cost_usd=0.13,
        ),
        hyperparameters={"epochs": "10", "batch_size": "32"},
    )
    body.update(overrides)
    return fleet.submit_training_job().execute(SubmitTrainingCommand(**body))  # type: ignore[arg-type]


def poll_training(fleet: FleetContainer, job_id: str = "train-2026-05-24", **overrides):  # noqa: ANN003, ANN201
    body: dict[str, object] = dict(job_id=job_id)
    body.update(overrides)
    return fleet.poll_training_job().execute(PollTrainingQuery(**body))  # type: ignore[arg-type]


def publish(fleet: FleetContainer, release: ReleaseBundle | None = None, **overrides):  # noqa: ANN003, ANN201
    body: dict[str, object] = dict(fleet_id=FLEET_ID, bundle=release or bundle())
    body.update(overrides)
    return fleet.publish_release().execute(PublishReleaseCommand(**body))  # type: ignore[arg-type]


def promote(fleet: FleetContainer, version: str, channel: ReleaseChannel):  # noqa: ANN201
    return fleet.promote_release().execute(
        PromoteReleaseCommand(fleet_id=FLEET_ID, version=version, channel=channel)
    )


def plan(fleet: FleetContainer, version: str = "v2.0.0", **overrides):  # noqa: ANN003, ANN201
    body: dict[str, object] = dict(
        fleet_id=FLEET_ID,
        rollout_id="ro-1",
        version=version,
        wave_sizes=(2, 8, 9999),
        group_order=GROUPS,
        occurred_at="2026-05-24 12:00:00",
    )
    body.update(overrides)
    return fleet.plan_rollout().execute(PlanRolloutCommand(**body))  # type: ignore[arg-type]


def collect(fleet: FleetContainer, rollout_id: str = "ro-1", **overrides):  # noqa: ANN003, ANN201
    body: dict[str, object] = dict(
        rollout_id=rollout_id, occurred_at="2026-05-24 12:30:00"
    )
    body.update(overrides)
    return fleet.collect_wave().execute(CollectWaveCommand(**body))  # type: ignore[arg-type]


def advance(fleet: FleetContainer, rollout_id: str = "ro-1", **overrides):  # noqa: ANN003, ANN201
    body: dict[str, object] = dict(
        rollout_id=rollout_id, fleet_id=FLEET_ID, occurred_at="2026-05-24 13:00:00"
    )
    body.update(overrides)
    return fleet.advance_rollout().execute(AdvanceRolloutCommand(**body))  # type: ignore[arg-type]


def apply_outcomes(fleet: FleetContainer, rollout_id: str = "ro-1", **overrides):  # noqa: ANN003, ANN201
    body: dict[str, object] = dict(
        fleet_id=FLEET_ID, rollout_id=rollout_id, seen_at="2026-05-24 13:30:00"
    )
    body.update(overrides)
    return fleet.apply_rollout_outcomes().execute(ApplyOutcomesCommand(**body))  # type: ignore[arg-type]


def rollback(fleet: FleetContainer, **overrides):  # noqa: ANN003, ANN201
    body: dict[str, object] = dict(
        fleet_id=FLEET_ID,
        rollout_id="ro-1",
        new_rollout_id="ro-1-rollback",
        reason="v2.0.0 배포 후 FAULT 재현율 붕괴",
        occurred_at="2026-05-25 09:00:00",
    )
    body.update(overrides)
    return fleet.rollback_rollout().execute(RollbackRolloutCommand(**body))  # type: ignore[arg-type]


def trace(fleet: FleetContainer, device_id: str = "DEV-00", **overrides):  # noqa: ANN003, ANN201
    body: dict[str, object] = dict(
        fleet_id=FLEET_ID,
        device_id=device_id,
        source_devices=tuple(f"DEV-{n:02d}" for n in range(DEVICE_COUNT)),
        window="2026-05-22 00:00:00 ~ 2026-05-23 23:59:59",
    )
    body.update(overrides)
    return fleet.trace_lineage().execute(TraceLineageQuery(**body))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 전체 파이프라인
# ---------------------------------------------------------------------------
def build_pipeline(*, with_rollout: bool = False):  # noqa: ANN201
    """플릿 생성 → 업링크 → 데이터셋 → 학습 → 릴리스 → 채널까지.

    롤아웃(6-8)은 기본으로 하지 않는다. 실습마다 다른 단계 계획을 써야 하기 때문이다.
    **호출하는 쪽이 moto 안에 있어야 한다.**
    """
    fleet = FleetContainer.with_aws(aws_config())
    fleet.ota.ensure_things([d.device_id for d in devices()])
    # 알림은 진짜 IoT Job 으로, 디바이스 응답은 흉내낸다 (moto 에 디바이스 API 가 없다).
    fleet.ota = SimulatedFleetOtaGateway(fleet.ota, response_profile())

    created = create(fleet)
    written = fill_lake(fleet)
    lake = inspect_lake(
        fleet,
        filters={"kind": "inference_log", "device": "DEV-02", "date": "2026-05-23"},
    )
    dataset = build_dataset(fleet)
    submitted = submit_training(fleet)
    polled = poll_training(fleet)
    published = publish(fleet)
    promote(fleet, "v2.0.0", ReleaseChannel.CANARY)

    result = SimpleNamespace(
        fleet=fleet,
        fleet_id=FLEET_ID,
        created=created,
        objects_written=written,
        lake=lake,
        dataset=dataset,
        submitted=submitted,
        polled=polled,
        published=published,
        rollout=None,
    )
    if with_rollout:
        result.rollout = plan(fleet)
    return result
