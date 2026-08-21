"""Fleet Aggregate 와 Fleet Context 의 Policy 들.

전부 손으로 만든 값으로 돌아간다. AWS 없이.
"""

from __future__ import annotations

import pytest

from domain.fleet.dataset_build import (
    DatasetBuildPolicy,
    DatasetBuildSpec,
    SourceWindow,
)
from domain.fleet.device import Device, DeviceStatus, FleetHealthPolicy
from domain.fleet.errors import DeviceNotFound, NotReleasable
from domain.fleet.fleet import Fleet
from domain.fleet.identifiers import FleetId
from domain.fleet.lineage import LineagePolicy, trace_of
from domain.fleet.object_key import KeyLayout, KeyLayoutPolicy, ObjectKey, ObjectStats
from domain.fleet.release import ReleaseBundle, ReleaseChannel, ReleasePolicy
from domain.fleet.training_job import (
    ComputeSpec,
    RemoteJobStatus,
    RemoteTrainingJob,
    TrainingBudgetPolicy,
)
from domain.fleet.uplink import UplinkBatch, UplinkKind, UplinkPolicy
from domain.shared.errors import IllegalStateTransition, InvariantViolation


def device(index: int, **overrides) -> Device:  # noqa: ANN003
    base: dict[str, object] = dict(
        device_id=f"DEV-{index:02d}",
        group="line-a",
        current_version="v1.0.0",
        last_seen_at="2026-05-23 09:00:00",
    )
    base.update(overrides)
    return Device(**base)  # type: ignore[arg-type]


def bundle(**overrides) -> ReleaseBundle:  # noqa: ANN003
    base: dict[str, object] = dict(
        release_id="r1",
        version="v2.0.0",
        model_version_id="mv-1",
        artifact_uri="s3://bucket/v2/model.tflite",
        artifact_bytes=11_724,
        checksum="c" * 32,
        runtime="TFLITE",
        precision="FP16",
        class_labels=("FAULT", "OVERLOAD", "NORMAL"),
        input_fields=("active_power_kw",),
        normalization={"active_power_kw": (147.8, 39.8)},
        expected_p95_ms=0.0031,
        expected_class_mix={"NORMAL": 0.78},
        sample_interval_seconds=10,
        window_length=30,
        source_build_id="build-1",
        source_job_id="train-1",
    )
    base.update(overrides)
    return ReleaseBundle(**base)  # type: ignore[arg-type]


def fleet_of(count: int = 6) -> Fleet:
    fleet = Fleet.create(FleetId.of("f1"), "3라인")
    fleet.register_many(device(n) for n in range(count))
    return fleet


class Test플릿:
    def test_이름_없는_플릿은_없다(self) -> None:
        with pytest.raises(InvariantViolation):
            Fleet.create(FleetId.of("f1"), "  ")

    def test_같은_식별자를_두_번_등록할_수_없다(self) -> None:
        fleet = fleet_of(1)
        with pytest.raises(InvariantViolation):
            fleet.register(device(0))

    def test_없는_디바이스는_없다고_말한다(self) -> None:
        with pytest.raises(DeviceNotFound):
            fleet_of(1).device("DEV-99")

    def test_그룹_없는_디바이스는_없다(self) -> None:
        with pytest.raises(InvariantViolation):
            Device(device_id="DEV-00", group="  ")

    def test_보고는_시간_순으로만(self) -> None:
        fleet = fleet_of(1)
        fleet.report("DEV-00", seen_at="2026-05-24 09:00:00")
        with pytest.raises(InvariantViolation):
            fleet.report("DEV-00", seen_at="2026-05-23 09:00:00")

    def test_버전_변경은_사건으로_남는다(self) -> None:
        fleet = fleet_of(1)
        fleet.pull_events()
        fleet.report("DEV-00", seen_at="2026-05-24 09:00:00", version="v2.0.0")
        assert [e.event_name for e in fleet.pull_events()] == ["DeviceVersionChanged"]

    def test_같은_버전_보고는_사건이_아니다(self) -> None:
        fleet = fleet_of(1)
        fleet.pull_events()
        fleet.report("DEV-00", seen_at="2026-05-24 09:00:00", version="v1.0.0")
        assert fleet.pull_events() == ()

    def test_격리된_디바이스는_배포_대상이_아니다(self) -> None:
        fleet = fleet_of(3)
        fleet.mark("DEV-01", DeviceStatus.QUARANTINED)
        assert {d.device_id for d in fleet.reachable_devices()} == {"DEV-00", "DEV-02"}

    def test_격리된_디바이스는_학습_대상도_아니다(self) -> None:
        fleet = fleet_of(3)
        fleet.mark("DEV-01", DeviceStatus.QUARANTINED)
        assert "DEV-01" not in {d.device_id for d in fleet.trainable_devices()}

    def test_훑기가_격리를_덮어쓰지_않는다(self) -> None:
        fleet = fleet_of(2)
        fleet.mark("DEV-00", DeviceStatus.QUARANTINED)
        fleet.sweep_stale(
            now="2026-05-30 00:00:00",
            stale_after="2026-05-29 00:00:00",
            unreachable_after="2026-05-28 00:00:00",
        )
        assert fleet.device("DEV-00").status is DeviceStatus.QUARANTINED
        assert fleet.device("DEV-01").status is DeviceStatus.UNREACHABLE

    def test_한_번도_보고_안_한_디바이스는_훑기_대상이_아니다(self) -> None:
        fleet = Fleet.create(FleetId.of("f1"), "x")
        fleet.register(device(0, last_seen_at=""))
        fleet.sweep_stale(
            now="2026-05-30 00:00:00",
            stale_after="2026-05-29 00:00:00",
            unreachable_after="2026-05-28 00:00:00",
        )
        assert fleet.device("DEV-00").status is DeviceStatus.HEALTHY
        assert fleet.summary.never_reported == 1

    def test_집계는_대수에_비례하지_않는다(self) -> None:
        summary = fleet_of(6).summary
        assert summary.total == 6
        assert summary.by_version == {"v1.0.0": 6}
        assert summary.dominant_share == 1.0


class Test릴리스와채널:
    def test_같은_버전을_두_번_등록할_수_없다(self) -> None:
        fleet = fleet_of(1)
        fleet.publish(bundle())
        with pytest.raises(InvariantViolation):
            fleet.publish(bundle(artifact_bytes=99))

    def test_등록되지_않은_버전은_승격할_수_없다(self) -> None:
        with pytest.raises(NotReleasable):
            fleet_of(1).promote("v9", ReleaseChannel.CANARY)

    def test_canary를_거치지_않으면_stable로_못_간다(self) -> None:
        fleet = fleet_of(1)
        fleet.publish(bundle())
        with pytest.raises(IllegalStateTransition):
            fleet.promote("v2.0.0", ReleaseChannel.STABLE)

    def test_한_채널에_하나뿐이다(self) -> None:
        fleet = fleet_of(1)
        fleet.publish(bundle())
        fleet.publish(bundle(version="v2.1.0", release_id="r2"))
        fleet.promote("v2.0.0", ReleaseChannel.CANARY)
        fleet.promote("v2.1.0", ReleaseChannel.CANARY)

        assert fleet.channels.canary == "v2.1.0"
        assert "v2.0.0" in fleet.channels.archived

    def test_계보가_함께_저장된다(self) -> None:
        fleet = fleet_of(1)
        fleet.publish(bundle())
        assert fleet.lineage_of("v2.0.0") == ("build-1", "train-1")
        assert fleet.lineage_of("없음") == ("", "")


class Test업링크:
    def test_시각이_거꾸로면_묶음이_아니다(self) -> None:
        with pytest.raises(InvariantViolation):
            UplinkBatch(
                device_id="DEV-00",
                kind=UplinkKind.INFERENCE_LOG,
                window_start="2026-05-23 10:00:00",
                window_end="2026-05-23 09:00:00",
                record_count=10,
                payload_bytes=100,
            )

    def test_금지된_열은_CRITICAL_이다(self) -> None:
        batch = UplinkBatch(
            device_id="DEV-00",
            kind=UplinkKind.RAW_SAMPLE,
            window_start="2026-05-23 09:00:00",
            window_end="2026-05-23 09:59:59",
            record_count=100,
            payload_bytes=10_000,
            checksum="x",
            fields=("phone",),
        )
        assert not UplinkPolicy().accepts(batch)

    def test_예산은_누적으로_본다(self) -> None:
        batch = UplinkBatch(
            device_id="DEV-00",
            kind=UplinkKind.INFERENCE_LOG,
            window_start="2026-05-23 09:00:00",
            window_end="2026-05-23 09:59:59",
            record_count=100,
            payload_bytes=100_000,
            checksum="x",
        )
        policy = UplinkPolicy(daily_budget_kib_per_device=150.0)
        assert policy.accepts(batch, sent_today_kib=0.0)
        assert not policy.accepts(batch, sent_today_kib=100.0)


class Test객체키:
    def test_접두어로_좁힌다(self) -> None:
        layout = KeyLayout()
        narrowed = layout.prefix_for(kind="inference_log", device="DEV-02")
        assert narrowed.endswith("device=DEV-02/")

    def test_중간이_비면_거기서_멈춘다(self) -> None:
        layout = KeyLayout()
        assert layout.prefix_for(kind="inference_log", date="2026-05-23").endswith(
            "kind=inference_log/"
        )

    def test_깊이로_잘라_쓸_수_있다(self) -> None:
        key = KeyLayout().key_for(
            kind="inference_log",
            device_id="DEV-02",
            date="2026-05-23",
            hour="09",
            part=1,
        )
        assert key.partition_prefix(2).endswith("device=DEV-02/")

    def test_작은_파일과_넓은_접두어를_둘_다_본다(self) -> None:
        findings = KeyLayoutPolicy().inspect(
            KeyLayout(),
            ObjectStats(
                object_count=100_000,
                total_bytes=100_000 * 2_000,
                distinct_prefixes=3,
            ),
        )
        codes = {f.code for f in findings}
        assert {"LAKE_SMALL_FILES", "LAKE_PREFIX_TOO_WIDE"} <= codes

    def test_키에_금지된_토큰이_들어갈_수_없다(self) -> None:
        with pytest.raises(InvariantViolation):
            ObjectKey(prefix="a", partitions=(), filename="b c")


class Test데이터셋:
    def spec(self, **overrides):  # noqa: ANN003, ANN201
        base: dict[str, object] = dict(
            build_id="b1",
            window=SourceWindow(started_at="a", ended_at="b"),
            device_ids=("DEV-00", "DEV-01"),
            record_counts={"DEV-00": 5000, "DEV-01": 5000},
            labeled_counts={"DEV-00": 400, "DEV-01": 400},
            label_distribution={"NORMAL": 600, "FAULT": 200},
        )
        base.update(overrides)
        return DatasetBuildSpec(**base)  # type: ignore[arg-type]

    def test_디바이스를_안_고르면_계획이_아니다(self) -> None:
        with pytest.raises(InvariantViolation):
            self.spec(device_ids=())

    def test_한_디바이스가_지배하면_경고한다(self) -> None:
        check = DatasetBuildPolicy().inspect(
            self.spec(record_counts={"DEV-00": 9_000, "DEV-01": 1_000})
        )
        assert "BUILD_DEVICE_DOMINATED" in {f.code for f in check.findings}

    def test_전부_통과하면_만들_수_있다(self) -> None:
        assert DatasetBuildPolicy().inspect(self.spec()).can_build


class Test원격학습:
    def test_결과를_어디_둘지_없으면_못_찾는다(self) -> None:
        with pytest.raises(InvariantViolation):
            RemoteTrainingJob(
                job_id="j",
                dataset_uri="s3://a/",
                output_uri="  ",
                compute=ComputeSpec(instance_type="ml.m5.large"),
            )

    def test_최악의_비용을_계산한다(self) -> None:
        compute = ComputeSpec(
            instance_type="ml.p3.2xlarge",
            instance_count=2,
            max_runtime_seconds=7_200,
            hourly_cost_usd=4.0,
        )
        assert compute.worst_case_cost_usd == pytest.approx(16.0)

    def test_1분짜리_학습은_학습이_아니다(self) -> None:
        with pytest.raises(InvariantViolation):
            ComputeSpec(instance_type="ml.m5.large", max_runtime_seconds=30)

    def test_끝난_상태만_판정한다(self) -> None:
        assert RemoteJobStatus.RUNNING.is_terminal is False
        assert RemoteJobStatus.STOPPED.is_terminal is True

    def test_지표가_기준_아래면_막는다(self) -> None:
        job = RemoteTrainingJob(
            job_id="j",
            dataset_uri="s3://a/",
            output_uri="s3://b/",
            compute=ComputeSpec(instance_type="ml.m5.large"),
            status=RemoteJobStatus.SUCCEEDED,
            artifact_uri="s3://b/model.tar.gz",
            metrics={"macro_recall": 0.5},
        )
        findings = TrainingBudgetPolicy(min_metrics={"macro_recall": 0.9}).inspect_result(
            job
        )
        assert "TRAIN_METRIC_BELOW_FLOOR" in {f.code for f in findings}


class Test릴리스정책:
    def test_대수를_곱하면_달라진다(self) -> None:
        check = ReleasePolicy(max_fleet_transfer_mib=10.0).inspect(
            bundle(artifact_bytes=100_000), device_count=200
        )
        assert "RELEASE_FLEET_TRANSFER_TOO_LARGE" in {f.code for f in check.findings}

    def test_전부_갖추면_내보낼_수_있다(self) -> None:
        assert ReleasePolicy().inspect(bundle(), device_count=24).can_publish


class Test플릿건강:
    def test_빈_플릿은_경고한다(self) -> None:
        findings = FleetHealthPolicy().inspect(fleet_of(0).summary)
        assert "FLEET_EMPTY" in {f.code for f in findings}


class Test계보:
    def test_다섯_칸이_다_있어야_닫힌다(self) -> None:
        trace = trace_of(
            device_id="DEV-00",
            version="v2",
            job_id="j",
            build_id="b",
            window="w",
        )
        assert LineagePolicy().inspect(trace, source_devices=("DEV-00",)).closed

    def test_빈_칸은_숨기지_않는다(self) -> None:
        trace = trace_of(
            device_id="DEV-00", version="", job_id="j", build_id="b", window="w"
        )
        assert trace.broken
        assert "(끊김)" in trace.render()
