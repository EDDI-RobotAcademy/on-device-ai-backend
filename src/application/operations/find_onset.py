"""AI가 언제부터 이상해졌는지 찾아라. (실습 5-4)

이 Use Case 는 아무것도 측정하지 않는다. **이미 남겨 둔 창들을 읽을 뿐이다.**
그래서 이 질문에 답할 수 있는지는 지금이 아니라 **배포하던 날** 결정됐다.
"""

from __future__ import annotations

from dataclasses import dataclass

from application.operations.dto import OnsetView, TimelineView, WatchView
from application.operations.support import load_deployment, load_watch, watch_for
from domain.operations.health import HealthMetric
from domain.operations.ports import DeploymentRepository, HealthWatchRepository


@dataclass(frozen=True, slots=True)
class FindOnsetQuery:
    watch_id: str
    metric: HealthMetric = HealthMetric.INPUT_PSI
    threshold: float = 0.2
    consecutive: int = 3
    """이만큼 연속으로 넘겨야 '무너진 것'이다. 한 번 튄 것과 구분한다."""


class FindOnset:
    def __init__(self, watches: HealthWatchRepository) -> None:
        self._watches = watches

    def execute(self, query: FindOnsetQuery) -> OnsetView:
        watch = load_watch(self._watches, query.watch_id)
        onset = watch.timeline.onset_of(
            query.metric, query.threshold, consecutive=query.consecutive
        )
        return OnsetView.of(str(watch.id), onset)


@dataclass(frozen=True, slots=True)
class GetTimelineQuery:
    watch_id: str


class GetTimeline:
    def __init__(self, watches: HealthWatchRepository) -> None:
        self._watches = watches

    def execute(self, query: GetTimelineQuery) -> TimelineView:
        return TimelineView.of(load_watch(self._watches, query.watch_id))


class GetWatch:
    def __init__(self, watches: HealthWatchRepository) -> None:
        self._watches = watches

    def execute(self, query: GetTimelineQuery) -> WatchView:
        return WatchView.of(load_watch(self._watches, query.watch_id))


@dataclass(frozen=True, slots=True)
class GetWatchForDeploymentQuery:
    deployment_id: str


class GetWatchForDeployment:
    def __init__(
        self, deployments: DeploymentRepository, watches: HealthWatchRepository
    ) -> None:
        self._deployments = deployments
        self._watches = watches

    def execute(self, query: GetWatchForDeploymentQuery) -> WatchView:
        deployment = load_deployment(self._deployments, query.deployment_id)
        return WatchView.of(watch_for(self._watches, deployment.id))


class ListWatches:
    def __init__(self, watches: HealthWatchRepository) -> None:
        self._watches = watches

    def execute(self) -> tuple[WatchView, ...]:
        return tuple(WatchView.of(w) for w in self._watches.list_all())
