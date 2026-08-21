"""Fleet Context 의 식별자."""

from __future__ import annotations

from dataclasses import dataclass

from domain.shared.identifier import Identifier


@dataclass(frozen=True, slots=True)
class FleetId(Identifier):
    """디바이스 집합 하나. 대개 라인이나 공장 단위다."""


@dataclass(frozen=True, slots=True)
class DeviceId(Identifier):
    """디바이스 한 대. **현장에서 물리적으로 만질 수 있는 것**이어야 한다."""


@dataclass(frozen=True, slots=True)
class RolloutId(Identifier):
    """한 번의 OTA 배포."""


@dataclass(frozen=True, slots=True)
class ReleaseId(Identifier):
    """디바이스로 내보낼 수 있게 묶인 결과물 하나."""


@dataclass(frozen=True, slots=True)
class TrainingJobId(Identifier):
    """클라우드에 맡긴 학습 한 번."""


@dataclass(frozen=True, slots=True)
class DatasetBuildId(Identifier):
    """현장 데이터로 만든 학습 데이터셋 하나."""
