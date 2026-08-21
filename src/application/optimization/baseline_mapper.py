"""Model → Optimization 번역기 (Anti-Corruption Layer).

Optimization Context 는 TrainingRun 을 모른다.
알아야 하는 것은 "무엇을 최적화하고, 원래 얼마나 좋았고, **승인은 받았는가**"뿐이다.

    TrainingRun (모듈 3)  →  BaselineModelRef
"""

from __future__ import annotations

from application.shared.errors import UnsupportedOperation
from domain.model.training_run import TrainingRun, TrainingStatus
from domain.optimization.baseline_ref import BaselineModelRef


def baseline_from(run: TrainingRun, *, split: str = "test") -> BaselineModelRef:
    """승인 판정을 받은 학습 결과를 최적화의 출발점으로 번역한다.

    승인받지 않았어도 **번역 자체는 된다.**
    막는 것은 OptimizationRun.start() 의 일이다 — 판단은 Domain 이 한다.
    """
    if run.model_version_id is None:
        raise UnsupportedOperation(
            "학습이 끝나지 않아 최적화할 모델이 없다.", subject=str(run.id)
        )
    evaluation = run.evaluation_of(split)
    if evaluation is None:
        raise UnsupportedOperation(
            f"'{split}' 평가가 없다. 원래 얼마나 좋았는지 모르면 비교할 수 없다.",
            subject=str(run.id),
        )
    profile = run.profile
    matrix = evaluation.matrix

    return BaselineModelRef(
        model_version_id=str(run.model_version_id),
        run_ref=str(run.id),
        input_shape=run.architecture.input_spec.shape,
        class_labels=matrix.labels,
        parameter_count=profile.parameter_count if profile else 0,
        mac_count=profile.mac_count if profile else 0,
        accuracy=matrix.accuracy,
        macro_recall=matrix.macro_recall,
        per_class_recall={
            label: matrix.recall_of(label) for label in matrix.labels
        },
        accepted=run.status is TrainingStatus.ACCEPTED,
    )
