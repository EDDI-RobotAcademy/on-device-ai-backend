"""MeasureResources / ScaleBatch — CPU·메모리와 배치를 실제로 재라. (실습 4-13, 4-14)

실습 4-5 는 **파일 크기**를 쟀다. 여기서는 **실행 중에 잡히는 것**을 잰다.
그 둘의 차이가 임베디드에서 배포 가능 여부를 가른다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data.dto import FindingView
from application.optimization.dto import BatchScalingView, ResourceUsageView
from application.optimization.support import load_run
from application.shared.errors import UnsupportedOperation
from domain.optimization.benchmark import MeasurementProtocol
from domain.optimization.ports import (
    BatchScalingMeter,
    OptimizationRunRepository,
    ResourceMeter,
)
from domain.optimization.resource import CycleTimePolicy, ResourcePolicy


@dataclass(frozen=True, slots=True)
class MeasureResourcesCommand:
    run_id: str
    labels: tuple[str, ...] = ()
    """빈 값이면 모든 결과물을 잰다."""

    protocol: MeasurementProtocol = MeasurementProtocol(
        warmup_runs=20, measured_runs=150
    )
    policy: ResourcePolicy = field(default_factory=ResourcePolicy)


class MeasureResources:
    def __init__(
        self, runs: OptimizationRunRepository, meter: ResourceMeter
    ) -> None:
        self._runs = runs
        self._meter = meter

    def execute(
        self, command: MeasureResourcesCommand
    ) -> tuple[ResourceUsageView, ...]:
        run = load_run(self._runs, command.run_id)
        candidates = _candidates(run, command.labels)

        views = []
        for candidate in candidates:
            usage = self._meter.measure(candidate.artifact, command.protocol)
            views.append(
                ResourceUsageView.of(
                    usage,
                    findings=tuple(
                        FindingView.of(f) for f in command.policy.inspect(usage)
                    ),
                )
            )
        return tuple(views)


@dataclass(frozen=True, slots=True)
class ScaleBatchCommand:
    run_id: str
    label: str
    batch_sizes: tuple[int, ...] = (1, 4, 16, 64)
    protocol: MeasurementProtocol = MeasurementProtocol(
        warmup_runs=20, measured_runs=120
    )
    cycle_time_ms: float = 30.0


class ScaleBatch:
    def __init__(
        self, runs: OptimizationRunRepository, meter: BatchScalingMeter
    ) -> None:
        self._runs = runs
        self._meter = meter

    def execute(self, command: ScaleBatchCommand) -> BatchScalingView:
        run = load_run(self._runs, command.run_id)
        candidate = next(
            (c for c in run.candidates if c.artifact.label == command.label), None
        )
        if candidate is None:
            raise UnsupportedOperation(
                f"'{command.label}' 이라는 결과물이 없다.", subject=command.run_id
            )

        scaling = self._meter.scale(
            candidate.artifact, command.batch_sizes, command.protocol
        )
        policy = CycleTimePolicy(cycle_time_ms=command.cycle_time_ms)
        return BatchScalingView.of(
            command.label,
            scaling,
            cycle_time_ms=command.cycle_time_ms,
            findings=tuple(FindingView.of(f) for f in policy.inspect(scaling)),
        )


def _candidates(run, labels: tuple[str, ...]):  # noqa: ANN001, ANN202
    everything = tuple(run.candidates)
    if not labels:
        return everything
    chosen = [c for c in everything if c.artifact.label in labels]
    if not chosen:
        raise UnsupportedOperation(
            f"요청한 결과물이 없다: {labels}", subject=str(run.id)
        )
    return chosen
