"""재학습이 필요한 순간을 직접 정의하라. (실습 5-11)

이 Use Case 가 모듈 5 의 마지막이자, 모듈 1 로 돌아가는 문이다.

    데이터 → 품질 → 모델 → 최적화 → 운영 → **다시 데이터**

그리고 돌아갈 때 들고 가는 것이 있다.
"무엇이 어떻게 변했는가"와 "어느 클래스의 라벨이 모자란가".
그 두 가지가 다음 데이터 수집 계획이 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.operations.dto import RetrainingView
from application.operations.support import commit, load_watch
from application.shared.ports import EventPublisher
from domain.operations.ports import HealthWatchRepository, InferenceLogStore
from domain.operations.retraining import LabelSupply, RetrainingPolicy
from domain.operations.window import ObservationWindow


@dataclass(frozen=True, slots=True)
class DecideRetrainingCommand:
    watch_id: str
    supply: LabelSupply | None = None
    """비우면 로그에서 직접 센다."""

    since: ObservationWindow | None = None
    measured_accuracy: float | None = None
    """정답이 붙은 표본에서 실제로 잰 정확도. 있으면 가장 강한 근거가 된다."""

    policy: RetrainingPolicy = field(default_factory=RetrainingPolicy)


class DecideRetraining:
    def __init__(
        self,
        watches: HealthWatchRepository,
        logs: InferenceLogStore,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._watches = watches
        self._logs = logs
        self._publisher = publisher

    def execute(self, command: DecideRetrainingCommand) -> RetrainingView:
        watch = load_watch(self._watches, command.watch_id)
        supply = command.supply or self._count_labels(watch, command.since)

        decision = command.policy.decide(
            watch.timeline,
            supply,
            measured_accuracy=command.measured_accuracy,
        )

        if decision.needed:
            watch.request_retraining(
                decision.urgency.value,
                tuple(reason.value for reason in decision.reasons),
            )
            commit(self._watches, watch, self._publisher)

        return RetrainingView.of(str(watch.id), decision)

    def _count_labels(
        self, watch, since: ObservationWindow | None
    ) -> LabelSupply:  # noqa: ANN001
        """로그에서 라벨 공급을 센다."""
        windows = list(self._logs.windows_of(watch.deployment_id))
        if since is not None:
            windows = [w for w in windows if w.started_at >= since.started_at]

        total = labeled = 0
        per_class: dict[str, int] = {}
        for window in windows:
            for record in self._logs.records_in(watch.deployment_id, window):
                total += 1
                if record.ground_truth is None:
                    continue
                labeled += 1
                per_class[record.ground_truth] = per_class.get(record.ground_truth, 0) + 1

        return LabelSupply(
            total_records=total,
            labeled_records=labeled,
            labeled_since_deploy=labeled,
            minority_label_counts=per_class,
        )
