"""Data Quality Context 가 바깥 세계에 요구하는 것(Port).

여섯 개의 측정기가 있다. 전부 "숫자만 센다".
합격/불합격을 말하는 측정기는 하나도 없다 — 그것은 Policy 의 일이다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from domain.data_quality.assessment import QualityAssessment
from domain.data_quality.balance import ClassBalanceMeasurement
from domain.data_quality.completeness import MissingValueMeasurement
from domain.data_quality.identifiers import AssessmentId
from domain.data_quality.label_quality import (
    LabelConsistencyRule,
    LabelErrorMeasurement,
)
from domain.data_quality.noise import NoiseMeasurement
from domain.data_quality.target import AssessmentTarget
from domain.data_quality.uniqueness import DuplicateMeasurement
from domain.data_quality.validity import OutlierMeasurement
from domain.data_quality.rebalancing import RebalancingOutcome, RebalancingPlan


@runtime_checkable
class QualityAssessmentRepository(Protocol):
    def save(self, assessment: QualityAssessment) -> None: ...

    def find_by_id(self, assessment_id: AssessmentId) -> QualityAssessment | None: ...

    def exists(self, assessment_id: AssessmentId) -> bool: ...

    def find_by_dataset(self, dataset_ref: str) -> Sequence[QualityAssessment]: ...

    def list_all(self) -> Sequence[QualityAssessment]: ...


@runtime_checkable
class MissingValueMeasurer(Protocol):
    """결측률, 연속 결측 길이, 결측 집중도, 은폐 결측 후보. (실습 2-2)"""

    def measure(self, target: AssessmentTarget) -> MissingValueMeasurement: ...


@runtime_checkable
class OutlierMeasurer(Protocol):
    """z-score / MAD / 물리범위 / 변화율 이상치. (실습 2-3)"""

    def measure(self, target: AssessmentTarget) -> OutlierMeasurement: ...


@runtime_checkable
class LabelErrorMeasurer(Protocol):
    """현장 규칙과 모순되는 라벨, 같은 입력에 다른 라벨. (실습 2-4)"""

    def measure(
        self, target: AssessmentTarget, rules: tuple[LabelConsistencyRule, ...]
    ) -> LabelErrorMeasurement: ...


@runtime_checkable
class ClassBalanceMeasurer(Protocol):
    """클래스별 표본 수. (실습 2-5)"""

    def measure(self, target: AssessmentTarget) -> ClassBalanceMeasurement: ...


@runtime_checkable
class NoiseMeasurer(Protocol):
    """SNR, 고주파 비중, 부호 반전 비율. (실습 2-6)"""

    def measure(self, target: AssessmentTarget) -> NoiseMeasurement: ...


@runtime_checkable
class DuplicateMeasurer(Protocol):
    """입력 기준 완전/근접 중복, 라벨 모순. (실습 2-7)"""

    def measure(self, target: AssessmentTarget) -> DuplicateMeasurement: ...


@runtime_checkable
class Resampler(Protocol):
    """불균형 완화 전략을 실제로 적용해 본다. (실습 2-11)

    **적용만 한다.** 무엇을 잃었는지는 RebalancingPolicy 가 판정한다.
    """

    def resample(
        self,
        uri: str,
        source_format: str,
        *,
        label_field: str,
        plan: RebalancingPlan,
        train_ratio: float = 0.7,
        seed: int = 42,
    ) -> RebalancingOutcome: ...
