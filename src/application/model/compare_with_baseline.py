"""CompareWithBaseline — 통계로 잡히는 이상을 AI라고 부르지 마라. (실습 3-13)

하는 일은 셋이다.
    1. 통계 검출기를 같은 데이터·같은 창으로 돌린다
    2. 학습 모델의 다중 분류 결과를 '이상 여부'로 접는다
    3. 둘을 나란히 놓고 Domain Policy 에게 "AI 를 쓸 근거가 있는가"를 묻는다

2번이 중요하다. 접지 않고 비교하면 공평하지 않다 —
통계 기반은 애초에 유형을 답하지 않기 때문이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data.dto import FindingView
from application.model.dto import BaselineComparisonView
from application.model.support import load_run
from application.shared.errors import UnsupportedOperation
from domain.model.evaluation import ConfusionMatrix
from domain.model.ports import BaselineDetector, TrainingRunRepository
from domain.model.statistical_baseline import (
    BaselineComparison,
    BaselineJustificationPolicy,
    DetectorSpec,
)

ANOMALY = "이상"
NORMAL = "정상"


@dataclass(frozen=True, slots=True)
class CompareWithBaselineCommand:
    run_id: str
    detector: DetectorSpec
    normal_label: str = "NORMAL"
    split: str = "test"
    policy: BaselineJustificationPolicy = field(
        default_factory=BaselineJustificationPolicy
    )


class CompareWithBaseline:
    def __init__(
        self,
        runs: TrainingRunRepository,
        detector: BaselineDetector,
    ) -> None:
        self._runs = runs
        self._detector = detector

    def execute(
        self, command: CompareWithBaselineCommand
    ) -> BaselineComparisonView:
        run = load_run(self._runs, command.run_id)
        if run.windowing is None:
            raise UnsupportedOperation(
                "창이 없는 학습(이미지)에는 시계열 통계 기준선을 붙일 수 없다.",
                subject=command.run_id,
            )
        result = run.evaluation_of(command.split)
        if result is None:
            raise UnsupportedOperation(
                f"'{command.split}' 평가가 없다. 비교할 대상이 없다.",
                subject=command.run_id,
            )

        statistical = self._detector.evaluate(
            run.data,
            command.detector,
            window_length=run.windowing.window_length,
            stride=run.windowing.stride,
            label_policy=run.windowing.label_policy,
            normal_label=command.normal_label,
            split=command.split,
        )
        collapsed = _collapse(result.matrix, normal_label=command.normal_label)

        anomaly_types = tuple(
            label for label in result.matrix.labels if label != command.normal_label
        )
        comparison = BaselineComparison(
            detector=command.detector.describe(),
            statistical_recall=statistical.matrix.recall_of(ANOMALY),
            statistical_precision=statistical.matrix.precision_of(ANOMALY),
            model_recall=collapsed.recall_of(ANOMALY),
            model_precision=collapsed.precision_of(ANOMALY),
            model_type_accuracy=_type_accuracy(
                result.matrix, normal_label=command.normal_label
            ),
            type_count=len(anomaly_types),
        )
        findings = command.policy.inspect(comparison)

        return BaselineComparisonView.of(
            command.run_id,
            comparison,
            statistical_matrix=statistical.matrix.render(),
            model_matrix=collapsed.render(),
            findings=tuple(FindingView.of(f) for f in findings),
        )


def _collapse(matrix: ConfusionMatrix, *, normal_label: str) -> ConfusionMatrix:
    """다중 분류 혼동 행렬을 '정상/이상' 2×2 로 접는다."""
    pairs: list[tuple[str, str]] = []
    for actual in matrix.labels:
        for predicted in matrix.labels:
            count = matrix.count_of(actual, predicted)
            if not count:
                continue
            pairs.extend(
                [
                    (
                        NORMAL if actual == normal_label else ANOMALY,
                        NORMAL if predicted == normal_label else ANOMALY,
                    )
                ]
                * count
            )
    return ConfusionMatrix.from_pairs((NORMAL, ANOMALY), pairs)


def _type_accuracy(matrix: ConfusionMatrix, *, normal_label: str) -> float:
    """실제로 이상이었던 것 중, **유형까지** 맞힌 비율.

    통계 기반은 이 숫자를 만들 수 없다. 유형이라는 개념이 없기 때문이다.
    """
    total = 0
    correct = 0
    for actual in matrix.labels:
        if actual == normal_label:
            continue
        for predicted in matrix.labels:
            count = matrix.count_of(actual, predicted)
            total += count
            if actual == predicted:
                correct += count
    return correct / total if total else 0.0
