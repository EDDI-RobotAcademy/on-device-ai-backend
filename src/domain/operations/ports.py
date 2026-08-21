"""Operations Context 가 바깥 세계에 요구하는 것(Port).

이 파일에 torch 도 pandas 도 boto3 도 없다.

    Domain          InferenceRecord / LatencyProfile / PredictionMix / DriftReport
        ↓ Port
    Infrastructure  파일 / DB / CloudWatch / 디바이스 에이전트 …

지금은 로그가 인메모리에 있다. 모듈 6 에서 S3 와 DynamoDB 로 간다.
바뀌는 것은 어댑터뿐이다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from domain.operations.deployment import Deployment
from domain.operations.drift import DriftReport
from domain.operations.identifiers import DeploymentId, WatchId
from domain.operations.inference_log import InferenceRecord
from domain.operations.latency import LatencyProfile
from domain.operations.prediction_mix import PredictionMix
from domain.operations.shadow import ShadowRun
from domain.operations.watch import HealthWatch
from domain.operations.window import ObservationWindow


@runtime_checkable
class DeploymentRepository(Protocol):
    def save(self, deployment: Deployment) -> None: ...

    def find_by_id(self, deployment_id: DeploymentId) -> Deployment | None: ...

    def exists(self, deployment_id: DeploymentId) -> bool: ...

    def list_all(self) -> Sequence[Deployment]: ...


@runtime_checkable
class HealthWatchRepository(Protocol):
    def save(self, watch: HealthWatch) -> None: ...

    def find_by_id(self, watch_id: WatchId) -> HealthWatch | None: ...

    def find_by_deployment(self, deployment_id: DeploymentId) -> HealthWatch | None: ...

    def exists(self, watch_id: WatchId) -> bool: ...

    def list_all(self) -> Sequence[HealthWatch]: ...


@runtime_checkable
class InferenceLogStore(Protocol):
    """AI의 모든 판단이 쌓이는 곳. (실습 5-3)

    현장에서는 디바이스가 여기로 밀어 넣는다. 모듈 6 에서 그 경로를 만든다.
    """

    def append(self, records: Sequence[InferenceRecord]) -> int: ...

    def records_in(
        self, deployment_id: DeploymentId, window: ObservationWindow
    ) -> Sequence[InferenceRecord]: ...

    def windows_of(self, deployment_id: DeploymentId) -> Sequence[ObservationWindow]: ...

    def attach_ground_truth(
        self, deployment_id: DeploymentId, digest: str, label: str
    ) -> bool:
        """나중에 사람이 정답을 붙인다. **대개 일부에만 붙는다.**"""
        ...


@runtime_checkable
class LatencyMeasurer(Protocol):
    """로그에서 지연시간 분포를 뽑는다. (실습 5-5)"""

    def measure(
        self, deployment_id: DeploymentId, window: ObservationWindow
    ) -> LatencyProfile: ...


@runtime_checkable
class PredictionMixMeasurer(Protocol):
    """로그에서 예측 구성을 센다. (실습 5-6)"""

    def measure(
        self, deployment_id: DeploymentId, window: ObservationWindow
    ) -> PredictionMix: ...


@runtime_checkable
class InputDriftMeasurer(Protocol):
    """현장 입력이 학습 때와 얼마나 다른지 잰다. (실습 5-7)

    이 어댑터만 원본 신호를 본다. 나머지는 전부 로그만 본다.
    """

    def measure(
        self, deployment_id: DeploymentId, window: ObservationWindow
    ) -> DriftReport: ...


@runtime_checkable
class ShadowRunner(Protocol):
    """같은 입력을 두 모델에 넣어 본다. (실습 5-9)"""

    def run(
        self,
        deployment_id: DeploymentId,
        window: ObservationWindow,
        candidate_artifact_id: str,
    ) -> ShadowRun: ...
