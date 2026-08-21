"""AWS 어댑터 — moto 로 **실제 boto3 호출**을 시험한다.

가짜 클라이언트를 세워 두고 "호출됐다"만 확인하면
API 이름이 틀려도, 파라미터가 빠져도 통과한다.
그건 어댑터를 시험한 것이 아니라 **자기가 쓴 mock 을 시험한 것**이다.

moto 는 boto3 요청을 가로채 메모리에서 처리한다.
`create_training_job` 의 필수 파라미터가 빠지면 여기서 터진다.
"""

from __future__ import annotations

import json

import pytest

from domain.fleet.device import Device, DeviceStatus
from domain.fleet.identifiers import DeviceId, FleetId, RolloutId
from domain.fleet.object_key import KeyLayout, ObjectKey
from domain.fleet.rollout import DeviceOutcome
from domain.fleet.training_job import ComputeSpec, RemoteJobStatus
from domain.fleet.uplink import UplinkBatch, UplinkKind
from infrastructure.aws.config import AwsConfig
from tests.support import fleet_scenario as fs

FLEET = FleetId.of("line3")


@pytest.fixture
def store(aws):  # noqa: ANN001, ANN201
    from infrastructure.aws.s3_object_store import S3ObjectStore

    adapter = S3ObjectStore(fs.aws_config())
    adapter.ensure_bucket()
    return adapter


@pytest.fixture
def registry(aws):  # noqa: ANN001, ANN201
    from infrastructure.aws.dynamo_device_registry import DynamoDeviceRegistry

    adapter = DynamoDeviceRegistry(fs.aws_config())
    adapter.ensure_tables()
    return adapter


@pytest.fixture
def training(aws):  # noqa: ANN001, ANN201
    from infrastructure.aws.sagemaker_training import SageMakerTrainingGateway

    return SageMakerTrainingGateway(fs.aws_config())


@pytest.fixture
def ota(aws):  # noqa: ANN001, ANN201
    from infrastructure.aws.iot_ota_gateway import IotJobsOtaGateway

    adapter = IotJobsOtaGateway(fs.aws_config())
    adapter.ensure_things([f"DEV-{n:02d}" for n in range(4)])
    return adapter


class TestS3ObjectStore:
    def key(self, hour: str = "09", part: int = 0) -> ObjectKey:
        return KeyLayout().key_for(
            kind="inference_log",
            device_id="DEV-02",
            date="2026-05-23",
            hour=hour,
            part=part,
        )

    def test_두고_다시_꺼낸다(self, store) -> None:
        uri = store.put(self.key(), b"hello")
        assert uri.startswith("s3://ondevice-ai-lake/")
        assert store.get(self.key()) == b"hello"

    def test_없는_객체는_없다고_말한다(self, store) -> None:
        from infrastructure.aws.s3_object_store import ObjectNotFound

        with pytest.raises(ObjectNotFound):
            store.get(self.key(hour="23"))

    def test_접두어로_좁혀_목록을_가져온다(self, store) -> None:
        for hour in ("09", "10"):
            for part in range(3):
                store.put(self.key(hour, part), b"x" * 100)

        narrow = KeyLayout().prefix_for(
            kind="inference_log", device="DEV-02", date="2026-05-23", hour="09"
        )
        assert len(store.list_prefix(narrow)) == 3
        assert len(store.list_prefix("uplinks/")) == 6

    def test_페이지가_넘어가도_전부_가져온다(self, store) -> None:
        """S3 목록은 1,000개씩 끊어 온다. paginator 를 안 쓰면 조용히 잘린다."""
        for part in range(1_200):
            store.put(self.key(part=part), b"x")
        assert len(store.list_prefix("uplinks/")) == 1_200

    def test_저장소가_어떻게_생겼는지_센다(self, store) -> None:
        for part in range(5):
            store.put(self.key(part=part), b"x" * 2_000)
        stats = store.stats("uplinks/")

        assert stats.object_count == 5
        assert stats.total_bytes == 10_000
        assert stats.distinct_prefixes == 1
        assert stats.mean_bytes == 2_000


class TestDynamoDeviceRegistry:
    def test_넣고_한_대를_찾는다(self, registry) -> None:
        registry.upsert(FLEET, Device(device_id="DEV-00", group="pilot"))
        found = registry.find(FLEET, DeviceId.of("DEV-00"))

        assert found is not None
        assert found.group == "pilot"
        assert found.status is DeviceStatus.HEALTHY

    def test_없는_디바이스는_None_이다(self, registry) -> None:
        assert registry.find(FLEET, DeviceId.of("없음")) is None

    def test_플릿_전체를_Query_로_가져온다(self, registry) -> None:
        for n in range(30):
            registry.upsert(FLEET, Device(device_id=f"DEV-{n:02d}", group="line-a"))
        registry.upsert(FleetId.of("other"), Device(device_id="X-01", group="g"))

        devices = registry.list_devices(FLEET)
        assert len(devices) == 30
        assert all(d.device_id.startswith("DEV-") for d in devices)

    def test_업링크는_원자적으로_누적된다(self, registry) -> None:
        """읽고-더하고-쓰면 동시에 올라온 두 묶음 중 하나가 사라진다."""
        batch = UplinkBatch(
            device_id="DEV-00",
            kind=UplinkKind.INFERENCE_LOG,
            window_start="2026-05-23 09:00:00",
            window_end="2026-05-23 09:59:59",
            record_count=100,
            payload_bytes=50_000,
        )
        for _ in range(4):
            registry.record_uplink(FLEET, batch)

        total = registry.uplink_bytes_today(FLEET, DeviceId.of("DEV-00"), "2026-05-23")
        assert total == 200_000

    def test_다른_날짜는_따로_센다(self, registry) -> None:
        assert registry.uplink_bytes_today(FLEET, DeviceId.of("DEV-00"), "2026-01-01") == 0


class TestSageMakerTrainingGateway:
    def submit(self, gateway, job_id: str = "j1"):  # noqa: ANN001, ANN201
        return gateway.submit(
            job_id,
            dataset_uri="s3://ondevice-ai-lake/datasets/b1/",
            output_uri="s3://ondevice-ai-artifacts/j1/",
            compute=ComputeSpec(
                instance_type="ml.m5.large",
                max_runtime_seconds=3_600,
                hourly_cost_usd=0.13,
            ),
            hyperparameters={"epochs": "10"},
        )

    def test_제출하면_작업이_생긴다(self, training) -> None:
        job = self.submit(training)
        assert job.job_id == "j1"
        assert job.dataset_uri.endswith("/b1/")
        assert job.output_uri.startswith("s3://")

    def test_비용_추정치는_우리_설정에서_온다(self, training) -> None:
        """AWS 응답에는 가격이 없다."""
        job = self.submit(training)
        assert job.compute.hourly_cost_usd == pytest.approx(0.13)
        assert job.compute.worst_case_cost_usd > 0

    def test_상태를_물어본다(self, training) -> None:
        self.submit(training)
        job = training.describe("j1")
        assert job.status.is_terminal or job.status is RemoteJobStatus.RUNNING

    def test_없는_작업은_없다고_말한다(self, training) -> None:
        from infrastructure.aws.sagemaker_training import TrainingJobUnavailable

        with pytest.raises(TrainingJobUnavailable):
            training.describe("없음")

    def test_moto_에는_stop_training_job_이_없다(self, training) -> None:
        """**가짜의 한계를 숨기지 않는다.**

        moto 는 `stop_training_job` 을 구현하지 않았다.
        그래서 이 경로는 여기서 시험되지 않는다 — 그 사실을 테스트로 적어 둔다.
        가짜로 시험할 때는 **무엇이 시험되지 않았는지**를 아는 것이 그만큼 중요하다.
        """
        self.submit(training, "j2")
        with pytest.raises(NotImplementedError):
            training.stop("j2", "실습 확인")

    def test_응답을_Domain_어휘로_번역한다(self) -> None:
        """어댑터의 본체는 이 번역이다."""
        from infrastructure.aws.sagemaker_training import _to_job

        failed = _to_job(
            {
                "TrainingJobName": "j3",
                "TrainingJobStatus": "Failed",
                "OutputDataConfig": {"S3OutputPath": "s3://b/out/"},
                "ResourceConfig": {"InstanceType": "ml.m5.large", "InstanceCount": 1},
                "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
            }
        )
        assert failed.status is RemoteJobStatus.FAILED
        # 이유가 응답에 없어도 Domain 불변식을 지켜서 넘긴다.
        assert "CloudWatch" in failed.failure_reason


class TestIotJobsOtaGateway:
    def test_실제_Job_과_문서가_만들어진다(self, ota) -> None:
        import boto3

        job_id = ota.announce(
            RolloutId.of("ro-1"), fs.bundle(), ["DEV-00", "DEV-01"]
        )
        iot = boto3.client("iot", region_name="ap-northeast-2")
        document = json.loads(iot.get_job_document(jobId=job_id)["document"])

        assert job_id == "ota-ro-1-v2-0-0"
        assert document["checksum"]
        assert document["sampleIntervalSeconds"] == 10
        assert document["normalization"]

    def test_버전_이름의_점은_안전한_문자로_바뀐다(self, ota) -> None:
        """IoT jobId 는 영숫자·대시·언더스코어만 받는다."""
        job_id = ota.announce(RolloutId.of("ro-2"), fs.bundle("v3.1.0"), ["DEV-00"])
        assert job_id == "ota-ro-2-v3-1-0"

    def test_응답이_없으면_PENDING_이다(self, ota) -> None:
        """**실패로 세지 않는다.**"""
        ota.announce(RolloutId.of("ro-3"), fs.bundle(), ["DEV-00", "DEV-01"])
        outcomes = ota.collect(RolloutId.of("ro-3"), ["DEV-00", "DEV-01"])
        assert set(outcomes.values()) == {DeviceOutcome.PENDING}

    def test_모르는_디바이스도_PENDING_이다(self, ota) -> None:
        outcomes = ota.collect(RolloutId.of("ro-4"), ["DEV-99"])
        assert outcomes["DEV-99"] is DeviceOutcome.PENDING

    def test_취소해도_터지지_않는다(self, ota) -> None:
        ota.announce(RolloutId.of("ro-5"), fs.bundle(), ["DEV-00"])
        ota.cancel(RolloutId.of("ro-5"), "롤백")
        ota.cancel(RolloutId.of("없음"), "없는 것 취소")


class TestSimulatedOta:
    def test_같은_디바이스는_같은_답을_한다(self, ota) -> None:
        """실패했던 디바이스가 다음 조회에서 성공하면 실패율이 거짓말을 한다."""
        from infrastructure.edge.ota_simulator import SimulatedFleetOtaGateway

        gateway = SimulatedFleetOtaGateway(ota, fs.response_profile())
        devices = [f"DEV-{n:02d}" for n in range(4)]
        first = gateway.collect(RolloutId.of("ro-1"), devices)
        second = gateway.collect(RolloutId.of("ro-1"), devices)
        assert first == second

    def test_못박은_결과가_우선한다(self, ota) -> None:
        from infrastructure.edge.ota_simulator import (
            FleetResponseProfile,
            SimulatedFleetOtaGateway,
        )

        gateway = SimulatedFleetOtaGateway(
            ota,
            FleetResponseProfile(forced={"DEV-00": DeviceOutcome.FAILED}),
        )
        assert (
            gateway.collect(RolloutId.of("ro-1"), ["DEV-00"])["DEV-00"]
            is DeviceOutcome.FAILED
        )

    def test_알리는_것은_진짜다(self, ota) -> None:
        import boto3

        from infrastructure.edge.ota_simulator import SimulatedFleetOtaGateway

        gateway = SimulatedFleetOtaGateway(ota)
        gateway.announce(RolloutId.of("ro-9"), fs.bundle(), ["DEV-00"])

        iot = boto3.client("iot", region_name="ap-northeast-2")
        jobs = [j["jobId"] for j in iot.list_jobs().get("jobs", [])]
        assert "ota-ro-9-v2-0-0" in jobs


class TestConfig:
    def test_로컬_대체_구현을_붙일_수_있다(self) -> None:
        """MinIO 나 LocalStack 을 쓸 때의 자리."""
        config = AwsConfig(endpoint_url="http://localhost:9000")
        assert config.client_kwargs()["endpoint_url"] == "http://localhost:9000"

    def test_기본은_엔드포인트를_안_넘긴다(self) -> None:
        assert "endpoint_url" not in AwsConfig().client_kwargs()
