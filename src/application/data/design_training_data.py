"""DesignTrainingData — 모델 입력 계약을 확정한다. (실습 1-7)

정규화 통계를 시스템이 직접 뽑아 주려면 분할이 먼저 있어야 한다.
그래서 이 Use Case 는 `fit_normalization=True` 일 때 Dataset 에 분할 계획이 있는지 확인한다.
없으면 진행하지 않는다 — "전체 데이터로 평균을 내면 되지 않나"라는 선택지를 아예 주지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from application.data.dto import InspectionView
from application.data.support import commit, load_dataset
from application.shared.errors import UnsupportedOperation
from application.shared.ports import EventPublisher
from domain.data.ports import DatasetRepository, NormalizationFitter
from domain.data.training_spec import (
    NormalizationMethod,
    NormalizationSpec,
    TrainingDataSpec,
)


@dataclass(frozen=True, slots=True)
class DesignTrainingDataCommand:
    dataset_id: str
    spec: TrainingDataSpec
    fit_normalization: bool = False
    """True 면 train 분할에서 정규화 통계를 직접 계산해 spec 에 채운다."""


@dataclass(frozen=True, slots=True)
class TrainingDesignView:
    dataset_id: str
    input_shape: tuple[int, ...]
    input_element_count: int
    feature_fields: tuple[str, ...]
    label_field: str
    window_seconds: float | None
    normalization_method: str
    normalization_fitted_on: str
    inspection: InspectionView

    def render(self) -> str:
        lines = [
            f"모델 입력 계약 ({self.dataset_id})",
            f"  input_shape     : {self.input_shape}  (원소 {self.input_element_count:,}개)",
            f"  feature_fields  : {', '.join(self.feature_fields)}",
            f"  label_field     : {self.label_field}",
            f"  normalization   : {self.normalization_method}"
            f" (통계 출처: {self.normalization_fitted_on})",
        ]
        if self.window_seconds is not None:
            lines.append(f"  window          : {self.window_seconds:g}초를 한 덩어리로")
        lines.append(self.inspection.render())
        return "\n".join(lines)


class DesignTrainingData:
    def __init__(
        self,
        repository: DatasetRepository,
        fitter: NormalizationFitter | None = None,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._repository = repository
        self._fitter = fitter
        self._publisher = publisher

    def execute(self, command: DesignTrainingDataCommand) -> TrainingDesignView:
        dataset = load_dataset(self._repository, command.dataset_id)
        spec = command.spec

        if command.fit_normalization:
            spec = self._fit(dataset, spec)

        report = dataset.design_training_data(spec)
        commit(self._repository, dataset, self._publisher)

        return TrainingDesignView(
            dataset_id=str(dataset.id),
            input_shape=spec.input_shape,
            input_element_count=spec.input_element_count,
            feature_fields=spec.feature_fields,
            label_field=spec.label_field,
            window_seconds=spec.window.duration_seconds if spec.window else None,
            normalization_method=spec.normalization.method.value,
            normalization_fitted_on=spec.normalization.fitted_on,
            inspection=InspectionView.of(str(dataset.id), report),
        )

    def _fit(self, dataset, spec: TrainingDataSpec) -> TrainingDataSpec:  # noqa: ANN001
        if self._fitter is None:
            raise UnsupportedOperation(
                "정규화 통계를 계산할 도구가 조립되어 있지 않다.", subject=str(dataset.id)
            )
        if spec.normalization.method is NormalizationMethod.NONE:
            raise UnsupportedOperation(
                "정규화 방식이 NONE 인데 통계를 계산하라고 했다.", subject="normalization"
            )
        if dataset.partition is None:
            raise UnsupportedOperation(
                "분할이 없다. 정규화 통계는 train 분할에서만 뽑을 수 있으므로 "
                "데이터를 먼저 나눠야 한다.",
                subject=str(dataset.id),
            )
        if dataset.schema is None:
            raise UnsupportedOperation("스키마가 없다.", subject=str(dataset.id))

        statistics = self._fitter.fit(
            dataset.source,
            dataset.schema,
            dataset.partition.plan,
            spec.feature_fields,
            spec.normalization.method,
        )
        normalization = NormalizationSpec(
            method=spec.normalization.method,
            fitted_on="train",
            statistics=dict(statistics),
        )
        return replace(spec, normalization=normalization)
