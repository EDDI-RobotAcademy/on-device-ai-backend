"""Fleet Context 가 바깥 세계에 요구하는 것(Port).

**이 파일에 boto3 도 S3 도 SageMaker 도 없다.** (CLAUDE.md §15)

    Domain          UplinkBatch / ObjectKey / RemoteTrainingJob / ReleaseBundle
        ↓ Port
    Infrastructure  S3ObjectStore / DynamoDeviceRegistry /
                    SageMakerTrainingGateway / IotJobsOtaGateway

AWS 를 다른 Cloud 로 바꾸면 바뀌는 것은 infrastructure/aws/ 하나다.
Port 이름에도 AWS 가 없다 — `ObjectStore` 이지 `S3Client` 가 아니다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from domain.fleet.device import Device
from domain.fleet.fleet import Fleet
from domain.fleet.identifiers import DeviceId, FleetId, RolloutId
from domain.fleet.object_key import ObjectKey, ObjectStats
from domain.fleet.release import ReleaseBundle
from domain.fleet.rollout import DeviceOutcome, Rollout
from domain.fleet.training_job import ComputeSpec, RemoteTrainingJob
from domain.fleet.uplink import UplinkBatch
from domain.fleet.endpoint import EndpointSpec, EndpointState
from domain.fleet.experiment_record import ExperimentLedger, ExperimentRecord
from domain.fleet.governance import AccessStatement, BucketGovernance


@runtime_checkable
class FleetRepository(Protocol):
    def save(self, fleet: Fleet) -> None: ...

    def find_by_id(self, fleet_id: FleetId) -> Fleet | None: ...

    def exists(self, fleet_id: FleetId) -> bool: ...

    def list_all(self) -> Sequence[Fleet]: ...


@runtime_checkable
class RolloutRepository(Protocol):
    def save(self, rollout: Rollout) -> None: ...

    def find_by_id(self, rollout_id: RolloutId) -> Rollout | None: ...

    def exists(self, rollout_id: RolloutId) -> bool: ...

    def list_all(self) -> Sequence[Rollout]: ...


@runtime_checkable
class ObjectStore(Protocol):
    """객체 저장소. (실습 6-1, 6-2)

    S3 라고 부르지 않는다. GCS 든 MinIO 든 같은 구멍이다.
    """

    def put(self, key: ObjectKey, body: bytes) -> str:
        """객체를 두고 그 위치(uri)를 돌려준다."""
        ...

    def get(self, key: ObjectKey) -> bytes: ...

    def list_prefix(self, prefix: str) -> Sequence[str]: ...

    def stats(self, prefix: str) -> ObjectStats:
        """이 접두어 아래가 어떻게 생겼는지 센다. (실습 6-2)"""
        ...


@runtime_checkable
class DeviceRegistry(Protocol):
    """디바이스 상태가 실제로 쌓이는 곳. (실습 6-3)

    Fleet Aggregate 는 판단을 하고, 이쪽은 수천 대를 담는다.
    """

    def upsert(self, fleet_id: FleetId, device: Device) -> None: ...

    def find(self, fleet_id: FleetId, device_id: DeviceId) -> Device | None: ...

    def list_devices(self, fleet_id: FleetId) -> Sequence[Device]: ...

    def record_uplink(self, fleet_id: FleetId, batch: UplinkBatch) -> None: ...

    def uplink_bytes_today(self, fleet_id: FleetId, device_id: DeviceId, date: str) -> int:
        """오늘 이 디바이스가 올린 양. 전송 예산 검사가 이것을 본다."""
        ...


@runtime_checkable
class RemoteTrainingGateway(Protocol):
    """남의 기계에서 학습시킨다. (실습 6-5)

    SageMaker 라고 부르지 않는다. Vertex AI 든 사내 Kubernetes 든 같은 모양이다.
    """

    def submit(
        self,
        job_id: str,
        *,
        dataset_uri: str,
        output_uri: str,
        compute: ComputeSpec,
        hyperparameters: dict[str, str] | None = None,
    ) -> RemoteTrainingJob: ...

    def describe(self, job_id: str) -> RemoteTrainingJob:
        """지금 어떻게 됐는지 물어본다. **기다리지 않는다** (CLAUDE.md §11)."""
        ...

    def stop(self, job_id: str, reason: str) -> RemoteTrainingJob: ...


@runtime_checkable
class OtaGateway(Protocol):
    """디바이스에 새 버전을 알린다. (실습 6-8, 6-9)

    IoT Jobs 라고 부르지 않는다. MQTT 든 HTTP 폴링이든 같은 일이다.
    """

    def announce(
        self, rollout_id: RolloutId, bundle: ReleaseBundle, device_ids: Sequence[str]
    ) -> str:
        """이 대상들에게 새 버전이 있다고 알린다. 작업 식별자를 돌려준다."""
        ...

    def collect(self, rollout_id: RolloutId, device_ids: Sequence[str]) -> dict[str, DeviceOutcome]:
        """디바이스들이 뭐라고 했는지 걷어 온다.

        **응답이 없는 것은 PENDING 이다.** 실패로 세지 않는다.
        """
        ...

    def cancel(self, rollout_id: RolloutId, reason: str) -> None: ...


@runtime_checkable
class ExperimentStore(Protocol):
    """실험 기록을 남기고 다시 읽는다. (실습 6-12)

    노트북이 아니라 **공용 저장소**에 둔다. 사람이 바뀌어도 남아야 하기 때문이다.
    """

    def record(self, entry: ExperimentRecord) -> str: ...

    def load(self, experiment_id: str) -> ExperimentLedger: ...


@runtime_checkable
class BucketGovernanceGateway(Protocol):
    """저장소의 버전·암호화·권한을 걸고 읽는다. (실습 6-13)

    이름에 S3 가 없다. 객체 저장소가 무엇이든 같은 문제가 있기 때문이다.
    """

    def harden(
        self,
        *,
        versioning: bool = True,
        encryption: str | None = "AES256",
        block_public_access: bool = True,
        expiration_days: int | None = 365,
    ) -> None: ...

    def put_policy(self, statements: tuple[AccessStatement, ...]) -> None: ...

    def inspect(self, *, version_prefix: str = "") -> BucketGovernance: ...


@runtime_checkable
class EndpointGateway(Protocol):
    """클라우드에 실시간 추론 자리를 띄운다. (실습 6-14)"""

    def deploy(self, spec: EndpointSpec, *, image_uri: str) -> EndpointState: ...

    def describe(self, name: str) -> EndpointState: ...

    def invoke(self, name: str, body: bytes) -> bytes: ...

    def teardown(self, name: str) -> None: ...
