"""Data Context 가 바깥 세계에 요구하는 것(Port).

Domain 은 "누가" 측정하는지 모른다. "무엇을 돌려줘야 하는지"만 안다.
pandas 를 polars 로, 로컬 파일을 S3 로 바꿔도 이 파일은 바뀌지 않는다.

구현은 전부 infrastructure/ 아래에 있다. (CLAUDE.md §8, §14, §15)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from domain.data.dataset import Dataset
from domain.data.identifiers import DatasetId
from domain.data.labeling import LabelAgreementMeasurement
from domain.data.partition import PartitionMeasurement, PartitionPlan
from domain.data.profile import DatasetProfile
from domain.data.representativeness import RepresentativenessMeasurement
from domain.data.sampling_design import SamplingObservation, SamplingPlan
from domain.data.schema import DataSchema
from domain.data.signal import ImageIntegrityMeasurement, SensorChannelMeasurement
from domain.data.source import DataSourceDescriptor
from domain.data.time_axis import TimeAxisMeasurement
from domain.data.training_spec import NormalizationMethod


@runtime_checkable
class DatasetRepository(Protocol):
    """Aggregate 의 영속화."""

    def save(self, dataset: Dataset) -> None: ...

    def find_by_id(self, dataset_id: DatasetId) -> Dataset | None: ...

    def exists(self, dataset_id: DatasetId) -> bool: ...

    def list_all(self) -> Sequence[Dataset]: ...


@runtime_checkable
class DatasetProfiler(Protocol):
    """데이터를 열어 사실을 관측한다. (실습 1-1)"""

    def profile(self, source: DataSourceDescriptor) -> DatasetProfile: ...


@runtime_checkable
class SchemaInferrer(Protocol):
    """관측으로부터 스키마 초안을 만든다. 확정은 사람이 한다. (실습 1-3)"""

    def infer(self, profile: DatasetProfile) -> DataSchema: ...


@runtime_checkable
class SensorSignalMeasurer(Protocol):
    """센서 채널의 고착/포화/범위 이탈을 센다. (실습 1-4)"""

    def measure(
        self, source: DataSourceDescriptor, schema: DataSchema
    ) -> tuple[SensorChannelMeasurement, ...]: ...


@runtime_checkable
class ImageSignalMeasurer(Protocol):
    """이미지의 판독 실패/초점/노출/중복을 센다. (실습 1-4)"""

    def measure(self, source: DataSourceDescriptor) -> ImageIntegrityMeasurement: ...


@runtime_checkable
class TimeAxisMeasurer(Protocol):
    """시간축의 역순/중복/공백을 센다. (실습 1-5)"""

    def measure(
        self, source: DataSourceDescriptor, time_field: str
    ) -> TimeAxisMeasurement: ...


@runtime_checkable
class LabelMeasurer(Protocol):
    """라벨 분포와 작업자 간 불일치를 센다. (실습 1-6)"""

    def measure(
        self, source: DataSourceDescriptor, label_field: str
    ) -> LabelAgreementMeasurement: ...


@runtime_checkable
class PartitionEngine(Protocol):
    """계획대로 실제로 나누고 누수를 센다. (실습 1-8)"""

    def apply(
        self,
        source: DataSourceDescriptor,
        schema: DataSchema,
        plan: PartitionPlan,
        label_field: str | None = None,
    ) -> PartitionMeasurement: ...


@runtime_checkable
class NormalizationFitter(Protocol):
    """정규화 통계를 train 분할에서만 계산한다. (실습 1-7)"""

    def fit(
        self,
        source: DataSourceDescriptor,
        schema: DataSchema,
        plan: PartitionPlan,
        feature_fields: tuple[str, ...],
        method: NormalizationMethod,
    ) -> Mapping[str, tuple[float, float]]: ...


@runtime_checkable
class DistributionComparer(Protocol):
    """학습 데이터와 현실 표본의 분포 차이를 잰다. (실습 1-9)"""

    def compare(
        self,
        reference: DataSourceDescriptor,
        observed: DataSourceDescriptor,
        schema: DataSchema,
    ) -> RepresentativenessMeasurement: ...


@runtime_checkable
class SamplingProbe(Protocol):
    """수집 주기를 바꿔 가며 원본을 다시 뽑아 본다. (실습 1-11)

    **재기만 한다.** 이 주기로 모아도 되는지는 SamplingDesignPolicy 가 정한다.
    """

    def probe(
        self,
        uri: str,
        source_format: str,
        *,
        time_field: str,
        label_field: str,
        normal_label: str,
        plan: SamplingPlan,
        value_field: str | None = None,
    ) -> SamplingObservation: ...
