"""PyTorch 학습 어댑터. (실습 3-5 ~ 3-8)

학습 루프는 여기 한 곳에만 있다.
Domain 은 결과로 나온 EpochRecord 만 받고, 그 숫자로 판단한다.

재현성:
    seed 를 고정하고, DataLoader 의 셔플도 같은 generator 를 쓴다.
    "어제는 됐는데요"를 없애기 위한 것이다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from domain.model.architecture import ModelArchitecture
from domain.model.curve import EpochRecord
from domain.model.evaluation import ConfusionMatrix, EvaluationResult
from domain.model.identifiers import ModelVersionId
from domain.model.protocol import SplitUsage
from domain.model.tensor_spec import BatchSpec
from domain.model.training_config import Optimizer, TrainingConfig
from domain.model.training_data_ref import TrainingDataRef
from domain.model.windowing import WindowLabelPolicy
from infrastructure.ml.torch_architecture import build_module
from infrastructure.ml.windowing import WindowedDataset, build_windows


@dataclass(slots=True)
class TorchTrainingOutcome:
    """domain.model.ports.TrainingOutcome 구현."""

    epochs: tuple[EpochRecord, ...]
    usage: SplitUsage
    artifact_uri: str
    boundary_overlap_samples: int = 0
    labels: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class TrainedModel:
    """학습이 끝난 뒤 남는 것. 평가와 배포가 이것을 쓴다.

    `dataset` 은 시계열이면 WindowedDataset, 이미지면 LabeledArrays 다.
    평가가 알아야 하는 것은 `features / targets / labels` 셋뿐이라
    둘을 구분하지 않는다.
    """

    module: nn.Module
    dataset: object
    architecture: ModelArchitecture
    best_state: dict[str, torch.Tensor]


class TorchModelRegistry:
    """학습된 모델을 들고 있는 곳.

    실제 시스템이라면 S3 나 파일시스템이다 (모듈 6).
    여기서는 메모리에 둔다 — 저장 위치가 바뀌어도 Domain 은 모른다.
    """

    def __init__(self) -> None:
        self._items: dict[str, TrainedModel] = {}

    def put(self, version_id: ModelVersionId, model: TrainedModel) -> None:
        self._items[str(version_id)] = model

    def get(self, version_id: ModelVersionId) -> TrainedModel | None:
        return self._items.get(str(version_id))

    def clear(self) -> None:
        self._items.clear()


class PyTorchModelTrainer:
    """domain.model.ports.ModelTrainer 구현."""

    def __init__(self, registry: TorchModelRegistry | None = None) -> None:
        self.registry = registry or TorchModelRegistry()
        self._last_version: ModelVersionId | None = None

    def train(
        self,
        data: TrainingDataRef,
        architecture: ModelArchitecture,
        config: TrainingConfig,
        batch: BatchSpec,
        window_length: int,
        stride: int,
        label_policy: WindowLabelPolicy,
    ) -> TorchTrainingOutcome:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.use_deterministic_algorithms(False)

        dataset = build_windows(
            data,
            window_length=window_length,
            stride=stride,
            label_policy=label_policy,
        )
        loaders = self._loaders(dataset, batch, config)

        module = build_module(architecture)
        criterion = self._criterion(dataset, config)
        optimizer = self._optimizer(module, config)

        records: list[EpochRecord] = []
        best_loss = float("inf")
        best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
        stale = 0

        for epoch in range(1, config.epochs + 1):
            started = time.perf_counter()
            train_loss, train_accuracy = self._run_epoch(
                module, loaders["train"], criterion, optimizer
            )
            validation_loss, validation_accuracy = self._run_epoch(
                module, loaders["validation"], criterion, None
            )
            records.append(
                EpochRecord(
                    epoch=epoch,
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                    train_accuracy=train_accuracy,
                    validation_accuracy=validation_accuracy,
                    duration_seconds=time.perf_counter() - started,
                )
            )

            # 가장 좋았던 가중치를 따로 보관한다. 마지막이 아니라 이것을 저장한다. (실습 3-7)
            if validation_loss < best_loss - (
                config.early_stopping.min_delta if config.early_stopping else 0.0
            ):
                best_loss = validation_loss
                best_state = {
                    k: v.detach().clone() for k, v in module.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if config.early_stopping and stale >= config.early_stopping.patience:
                    break

        version_id = ModelVersionId.of(
            f"mv-{data.dataset_ref}-{architecture.kind.value.lower()}-s{config.seed}"
        )
        self.registry.put(
            version_id,
            TrainedModel(
                module=module,
                dataset=dataset,
                architecture=architecture,
                best_state=best_state,
            ),
        )
        self._last_version = version_id

        return TorchTrainingOutcome(
            epochs=tuple(records),
            usage=SplitUsage(
                train_sample_count=int(dataset.features["train"].shape[0]),
                validation_sample_count=int(dataset.features["validation"].shape[0]),
                test_sample_count=int(dataset.features["test"].shape[0]),
                validation_evaluations=len(records),
                test_evaluations=0,
                overlapping_samples=dataset.boundary_overlap_samples,
            ),
            artifact_uri=f"memory://models/{version_id}",
            boundary_overlap_samples=dataset.boundary_overlap_samples,
            labels=dataset.labels,
        )

    @property
    def last_version_id(self) -> ModelVersionId | None:
        return self._last_version

    # -- 내부 --------------------------------------------------------------
    def _loaders(
        self, dataset: WindowedDataset, batch: BatchSpec, config: TrainingConfig
    ) -> dict[str, DataLoader]:
        return build_loaders(dataset, batch, config)

    def _criterion(
        self, dataset: WindowedDataset, config: TrainingConfig
    ) -> nn.Module:
        return build_criterion(dataset, config)

    def _optimizer(
        self, module: nn.Module, config: TrainingConfig
    ) -> torch.optim.Optimizer:
        return build_optimizer(module, config)

    def _run_epoch(
        self,
        module: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None,
    ) -> tuple[float, float]:
        return run_epoch(module, loader, criterion, optimizer)


# ---------------------------------------------------------------------------
# 학습 루프의 부품들
#
# 시계열이든 이미지든 학습 루프는 같다. 다른 것은 재료뿐이다.
# 그래서 여기 한 곳에 두고, 두 어댑터(시계열·이미지)가 함께 쓴다.
# ---------------------------------------------------------------------------
def build_loaders(
    dataset: object, batch: BatchSpec, config: TrainingConfig
) -> dict[str, DataLoader]:
    generator = torch.Generator().manual_seed(config.seed)
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "validation", "test"):
        tensors = TensorDataset(
            torch.from_numpy(dataset.features[split]),  # type: ignore[attr-defined]
            torch.from_numpy(dataset.targets[split]),  # type: ignore[attr-defined]
        )
        loaders[split] = DataLoader(
            tensors,
            batch_size=batch.batch_size,
            shuffle=batch.shuffle and split == "train",
            drop_last=batch.drop_last and split == "train",
            generator=generator if split == "train" else None,
        )
    return loaders


def build_criterion(dataset: object, config: TrainingConfig) -> nn.Module:
    """손실 함수. 가중치를 걸면 적은 클래스의 실수가 더 아프게 계산된다. (실습 2-11)"""
    if not config.class_weighted_loss:
        return nn.CrossEntropyLoss()
    targets = dataset.targets["train"]  # type: ignore[attr-defined]
    counts = np.bincount(
        targets, minlength=dataset.class_count  # type: ignore[attr-defined]
    ).astype("float64")
    weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))


def build_optimizer(
    module: nn.Module, config: TrainingConfig
) -> torch.optim.Optimizer:
    if config.optimizer is Optimizer.SGD:
        return torch.optim.SGD(
            module.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    return torch.optim.Adam(
        module.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def run_epoch(
    module: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    training = optimizer is not None
    module.train(training)

    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for inputs, targets in loader:
            if training:
                optimizer.zero_grad()
            logits = module(inputs)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * targets.size(0)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            seen += int(targets.size(0))

    if seen == 0:
        return 0.0, 0.0
    return total_loss / seen, correct / seen


class TorchModelEvaluator:
    """domain.model.ports.ModelEvaluator 구현. (실습 3-9)

    혼동 행렬과 지연시간을 함께 잰다.
    지연시간은 모듈 4 의 출발점이다 — 정확도만 재는 평가는 절반짜리다.
    """

    def __init__(
        self, registry: TorchModelRegistry, latency_samples: int = 64
    ) -> None:
        self._registry = registry
        self._latency_samples = latency_samples

    def evaluate(
        self, model_version_id: ModelVersionId, split: str
    ) -> EvaluationResult:
        trained = self._registry.get(model_version_id)
        if trained is None:
            raise KeyError(f"학습된 모델이 없다: {model_version_id}")

        module = trained.module
        module.load_state_dict(trained.best_state)  # 마지막이 아니라 최고 지점
        module.eval()

        features = torch.from_numpy(trained.dataset.features[split])
        targets = trained.dataset.targets[split]

        with torch.no_grad():
            logits = module(features)
            loss = float(
                nn.CrossEntropyLoss()(logits, torch.from_numpy(targets)).item()
            )
            predictions = logits.argmax(dim=1).numpy()

        labels = trained.dataset.labels
        matrix = ConfusionMatrix.from_pairs(
            labels,
            [
                (labels[int(actual)], labels[int(predicted)])
                for actual, predicted in zip(targets, predictions, strict=True)
            ],
        )
        p50, p95 = self._latency(module, features)
        return EvaluationResult(
            split=split,
            matrix=matrix,
            loss=loss,
            latency_ms_p50=p50,
            latency_ms_p95=p95,
        )

    def evaluate_external(
        self,
        model_version_id: ModelVersionId,
        data: TrainingDataRef,
        *,
        window_length: int,
        stride: int,
        label_policy: WindowLabelPolicy,
        split_name: str = "field",
    ) -> EvaluationResult:
        """학습에 쓰이지 않은 **다른 파일**로 평가한다. (실습 3-10)

        test 분할은 같은 날, 같은 조건에서 나온다.
        "현장에서 살아남는가"는 다른 날 데이터로만 답할 수 있다.
        """
        trained = self._registry.get(model_version_id)
        if trained is None:
            raise KeyError(f"학습된 모델이 없다: {model_version_id}")

        holdout = build_windows(
            data,
            window_length=window_length,
            stride=stride,
            label_policy=label_policy,
        )
        if holdout.labels != trained.dataset.labels:
            raise ValueError(
                "홀드아웃의 라벨 순서가 학습 때와 다르다. 혼동 행렬을 비교할 수 없다."
            )

        features = torch.from_numpy(
            np.concatenate([holdout.features[s] for s in ("train", "validation", "test")])
        )
        targets = np.concatenate(
            [holdout.targets[s] for s in ("train", "validation", "test")]
        )

        module = trained.module
        module.load_state_dict(trained.best_state)
        module.eval()
        with torch.no_grad():
            logits = module(features)
            loss = float(
                nn.CrossEntropyLoss()(logits, torch.from_numpy(targets)).item()
            )
            predictions = logits.argmax(dim=1).numpy()

        labels = holdout.labels
        matrix = ConfusionMatrix.from_pairs(
            labels,
            [
                (labels[int(actual)], labels[int(predicted)])
                for actual, predicted in zip(targets, predictions, strict=True)
            ],
        )
        p50, p95 = self._latency(module, features)
        return EvaluationResult(
            split=split_name,
            matrix=matrix,
            loss=loss,
            latency_ms_p50=p50,
            latency_ms_p95=p95,
        )

    def _latency(self, module: nn.Module, features: torch.Tensor) -> tuple[float, float]:
        """배치 1개짜리 추론 시간. 디바이스에서 재는 것과 같은 방식이다."""
        if features.shape[0] == 0:
            return 0.0, 0.0
        sample = features[:1]
        with torch.no_grad():
            for _ in range(5):  # 워밍업 — 첫 호출은 항상 느리다
                module(sample)
            timings = []
            for _ in range(self._latency_samples):
                started = time.perf_counter()
                module(sample)
                timings.append((time.perf_counter() - started) * 1000.0)
        array = np.array(timings)
        return float(np.percentile(array, 50)), float(np.percentile(array, 95))
