"""CompareQuantization — 학습 중에 양자화를 가르쳐라. (실습 4-12)

PTQ 와 QAT 를 **같은 비트 폭, 같은 평가 집합**으로 나란히 잰다.
그리고 Domain 에게 묻는다.

    "이 모델에 QAT 를 도입할 근거가 있는가?"

정확도가 더 높다는 것만으로는 근거가 아니다.
학습 파이프라인 하나를 영구히 더 유지하는 비용과 견줘야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data.dto import FindingView
from application.optimization.dto import QuantizationComparisonView
from application.optimization.support import load_run
from domain.optimization.ports import OptimizationRunRepository, QuantizationLab
from domain.optimization.quantization import QuantizationPolicy


@dataclass(frozen=True, slots=True)
class CompareQuantizationCommand:
    run_id: str
    bits: int = 8
    split: str = "test"
    epochs: int = 12
    per_channel: bool = True
    policy: QuantizationPolicy = field(default_factory=QuantizationPolicy)


class CompareQuantization:
    def __init__(
        self, runs: OptimizationRunRepository, lab: QuantizationLab
    ) -> None:
        self._runs = runs
        self._lab = lab

    def execute(
        self, command: CompareQuantizationCommand
    ) -> QuantizationComparisonView:
        run = load_run(self._runs, command.run_id)
        comparison = self._lab.compare(
            run.baseline,
            bits=command.bits,
            split=command.split,
            epochs=command.epochs,
            per_channel=command.per_channel,
        )
        findings = command.policy.inspect(comparison)
        return QuantizationComparisonView.of(
            command.run_id,
            comparison,
            findings=tuple(FindingView.of(f) for f in findings),
        )
