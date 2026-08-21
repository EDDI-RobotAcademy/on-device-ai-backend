"""Model Context 가 바깥 세계에 요구하는 것(Port).

이 파일에 PyTorch 라는 단어는 없다. 있어서도 안 된다. (CLAUDE.md §14)

    Domain          ModelArchitecture / TrainingCurve / ConfusionMatrix
        ↓ Port
    Infrastructure  PyTorchModelTrainer / TorchTensorMaterializer …

프레임워크를 바꾸면 어댑터만 새로 쓴다. 위쪽은 그대로다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from domain.model.architecture import ArchitectureProfile, ModelArchitecture
from domain.model.curve import EpochRecord
from domain.model.evaluation import EvaluationResult
from domain.model.identifiers import ModelVersionId, TrainingRunId
from domain.model.image_data_ref import ImageDataRef, ImageFolderReport
from domain.model.protocol import SplitUsage
from domain.model.statistical_baseline import DetectorSpec
from domain.model.tensor_spec import BatchSpec, DatasetTensorSummary, ImageTensorSpec
from domain.model.training_config import TrainingConfig
from domain.model.training_data_ref import TrainingDataRef
from domain.model.training_run import TrainingRun
from domain.model.windowing import WindowLabelPolicy, WindowingSummary


@runtime_checkable
class TrainingRunRepository(Protocol):
    def save(self, run: TrainingRun) -> None: ...

    def find_by_id(self, run_id: TrainingRunId) -> TrainingRun | None: ...

    def exists(self, run_id: TrainingRunId) -> bool: ...

    def list_all(self) -> Sequence[TrainingRun]: ...


@runtime_checkable
class TensorMaterializer(Protocol):
    """CSV 를 텐서로 바꾼다. (실습 3-1, 3-4)

    돌려주는 것은 텐서가 아니라 **텐서에 대한 요약**이다.
    실제 배열은 Infrastructure 안에 캐시되고, Domain 은 모양과 통계만 본다.
    """

    def materialize(
        self,
        data: TrainingDataRef,
        window_length: int,
        stride: int,
        label_policy: WindowLabelPolicy,
    ) -> tuple[WindowingSummary, dict[str, DatasetTensorSummary]]: ...


@runtime_checkable
class ImageTensorMaterializer(Protocol):
    """이미지를 텐서로 바꾼다. (실습 3-3)"""

    def materialize(
        self, root_uri: str, spec: ImageTensorSpec
    ) -> dict[str, DatasetTensorSummary]: ...


@runtime_checkable
class ImageFolderInspector(Protocol):
    """이미지 폴더를 열어 장수·해상도·중복을 센다. (실습 3-11)

    **재기만 한다.** 이 숫자로 학습을 시작해도 되는지는
    ImageReadinessPolicy 가 정한다 — 측정은 Infrastructure, 판정은 Domain.
    """

    def inspect(self, root_uri: str) -> ImageFolderReport: ...


@runtime_checkable
class ImageModelTrainer(Protocol):
    """이미지로 학습시킨다. (실습 3-11)

    `ModelTrainer` 와 나란히 둔 이유는 재료가 다르기 때문이다.
    창 길이도 stride 도 없다 — 이미지에는 자를 시간 축이 없다.
    """

    def train(
        self,
        data: ImageDataRef,
        architecture: ModelArchitecture,
        config: TrainingConfig,
        batch: BatchSpec,
    ) -> TrainingOutcome: ...


@runtime_checkable
class ArchitectureProfiler(Protocol):
    """구조를 조립해 층별 모양·파라미터·연산량을 센다. (실습 3-2)"""

    def profile(self, architecture: ModelArchitecture) -> ArchitectureProfile: ...


@runtime_checkable
class TrainingOutcome(Protocol):
    """학습이 남긴 것."""

    @property
    def epochs(self) -> tuple[EpochRecord, ...]: ...

    @property
    def usage(self) -> SplitUsage: ...

    @property
    def artifact_uri(self) -> str: ...


@runtime_checkable
class ModelTrainer(Protocol):
    """실제로 학습시킨다. (실습 3-5 ~ 3-8)"""

    def train(
        self,
        data: TrainingDataRef,
        architecture: ModelArchitecture,
        config: TrainingConfig,
        batch: BatchSpec,
        window_length: int,
        stride: int,
        label_policy: WindowLabelPolicy,
    ) -> TrainingOutcome: ...


@runtime_checkable
class BaselineDetector(Protocol):
    """학습 없이 '평소와 다르다'를 판정한다. (실습 3-13)

    AI 를 쓰기 전에 이것부터 재 본다.
    이기지 못하면 신경망을 얹을 이유가 없다.
    """

    def evaluate(
        self,
        data: TrainingDataRef,
        spec: DetectorSpec,
        *,
        window_length: int,
        stride: int,
        label_policy: WindowLabelPolicy,
        normal_label: str,
        split: str = "test",
    ) -> EvaluationResult: ...


@runtime_checkable
class ModelEvaluator(Protocol):
    """학습된 모델로 한 분할을 평가한다. (실습 3-9)"""

    def evaluate(
        self, model_version_id: ModelVersionId, split: str
    ) -> EvaluationResult: ...


@runtime_checkable
class FieldEvaluator(Protocol):
    """학습에 쓰이지 않은 **다른 파일**로 평가한다. (실습 3-10)

    test 분할은 같은 날, 같은 조건에서 나온다.
    "현장에서 살아남는가"는 다른 날 데이터로만 답할 수 있다.
    """

    def evaluate_external(
        self,
        model_version_id: ModelVersionId,
        data: TrainingDataRef,
        *,
        window_length: int,
        stride: int,
        label_policy: WindowLabelPolicy,
        split_name: str = "field",
    ) -> EvaluationResult: ...
