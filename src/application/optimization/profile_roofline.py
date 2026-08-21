"""ProfileRoofline — Latency는 연산량만으로 결정되지 않는다. (실습 4-9)

층마다 두 숫자를 센다: 계산량(MAC)과 옮긴 바이트.
둘의 비가 기계의 균형점보다 낮으면, 그 층은 계산이 아니라 **메모리에 묶여 있다.**

그런 층에서는 연산을 줄여도 빨라지지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data.dto import FindingView
from application.optimization.dto import RooflineView
from application.optimization.support import commit, load_run
from application.shared.ports import EventPublisher
from domain.optimization.ports import OptimizationRunRepository, RooflineProfiler
from domain.optimization.roofline import DeviceCapability, RooflinePolicy


@dataclass(frozen=True, slots=True)
class ProfileRooflineCommand:
    run_id: str
    device: DeviceCapability
    policy: RooflinePolicy = field(default_factory=RooflinePolicy)


class ProfileRoofline:
    def __init__(
        self,
        runs: OptimizationRunRepository,
        profiler: RooflineProfiler,
        publisher: EventPublisher | None = None,
    ) -> None:
        self._runs = runs
        self._profiler = profiler
        self._publisher = publisher

    def execute(self, command: ProfileRooflineCommand) -> RooflineView:
        run = load_run(self._runs, command.run_id)
        profile = self._profiler.profile(run.baseline, command.device)
        run.attach_roofline(profile)
        commit(self._runs, run, self._publisher)

        findings = command.policy.inspect(profile)
        return RooflineView.of(
            str(run.id), profile, tuple(FindingView.of(f) for f in findings)
        )
