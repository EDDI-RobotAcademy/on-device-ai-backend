"""Operations Context 의 식별자."""

from __future__ import annotations

from dataclasses import dataclass

from domain.shared.identifier import Identifier


@dataclass(frozen=True, slots=True)
class DeploymentId(Identifier):
    """한 대상(디바이스/그룹/플릿)에 대한 배포 하나."""


@dataclass(frozen=True, slots=True)
class WatchId(Identifier):
    """한 배포를 지켜보는 관측 기록."""


@dataclass(frozen=True, slots=True)
class IncidentId(Identifier):
    """현장에서 실제로 벌어진 사건 하나."""
