"""실습 3-9 — Accuracy 뒤에 숨어 있는 실패를 찾아라.

    pytest -m lesson_3_9 -s

정확도 97.4% 짜리 모델을 만들었다. 보고서에 쓰기 좋은 숫자다.
그 숫자 뒤를 본다.
"""

from __future__ import annotations

import pytest

from domain.model.evaluation import ConfusionMatrix, EvaluationPolicy, EvaluationResult
from domain.shared.inspection import Severity
from tests.support import model_scenario as ms
from tests.support import report

pytestmark = pytest.mark.lesson_3_9

LABELS = ("FAULT", "OVERLOAD", "NORMAL")


def codes(findings) -> set[str]:  # noqa: ANN001
    return {f.code for f in findings}


def test_정확도_뒤에는_클래스별_성적표가_있다(trained) -> None:
    report.section("실습 3-9 · Accuracy 뒤에 숨어 있는 실패를 찾아라")

    view = trained.evaluation
    report.block("평가 결과", view.render())

    assert view.accuracy > 0.95
    fault_recall = view.recall_of("FAULT")
    report.note(
        f"정확도 {view.accuracy:.1%} — 좋아 보인다. "
        f"그런데 FAULT 재현율은 {fault_recall:.1%} 다."
    )
    report.note(
        "설비 정지 8건 중 몇 건을 놓쳤다는 뜻이다. "
        "그 몇 건이 이 시스템을 만든 이유였다."
    )
    assert fault_recall < 1.0 or view.accuracy < 1.0


def test_한_번도_예측하지_않은_클래스는_없는_것과_같다() -> None:
    """가장 흔한 실패 — 소수 클래스를 아예 출력하지 않는 모델."""
    lazy = ConfusionMatrix.from_pairs(
        LABELS,
        [("FAULT", "NORMAL")] * 8
        + [("OVERLOAD", "OVERLOAD")] * 42
        + [("NORMAL", "NORMAL")] * 145,
    )
    result = EvaluationResult(split="test", matrix=lazy)
    findings = EvaluationPolicy(critical_labels=frozenset({"FAULT"})).inspect(result)

    report.block("FAULT 를 한 번도 예측하지 않은 모델", lazy.render())

    assert lazy.accuracy == pytest.approx(187 / 195)
    assert lazy.never_predicted == ("FAULT",)
    assert "EVAL_CLASS_NEVER_PREDICTED" in codes(findings)
    report.note(
        f"정확도 {lazy.accuracy:.1%}. baseline({lazy.baseline_accuracy:.1%})보다 높다. "
        "그런데 이 모델은 FAULT 에 대해서는 존재하지 않는 것과 같다."
    )
    report.note(
        f"macro recall 은 {lazy.macro_recall:.3f} 다. "
        "클래스마다 같은 무게로 세면 실패가 바로 보인다."
    )


def test_정확도가_baseline_을_못_넘으면_막는다() -> None:
    useless = ConfusionMatrix.from_pairs(
        LABELS,
        [("FAULT", "NORMAL")] * 8
        + [("OVERLOAD", "NORMAL")] * 42
        + [("NORMAL", "NORMAL")] * 145,
    )
    findings = EvaluationPolicy().inspect(
        EvaluationResult(split="test", matrix=useless)
    )
    assert "EVAL_NO_BETTER_THAN_BASELINE" in codes(findings)
    report.note(
        f"전부 NORMAL 이라고 찍은 모델. 정확도 {useless.accuracy:.1%}. "
        "학습을 안 한 것과 결과가 같다."
    )


def test_놓치면_안_되는_클래스는_기준이_더_세다(trained) -> None:
    """현장이 정하는 것이다. 트립을 놓치는 것과 과부하를 놓치는 것은 무게가 다르다."""
    from application.model.evaluate_model import EvaluateModelCommand

    view = trained.model.evaluate_model().execute(
        EvaluateModelCommand(
            run_id=trained.run_id,
            split="validation",
            policy=EvaluationPolicy(
                critical_labels=frozenset({"FAULT"}),
                min_critical_recall=0.95,
                min_recall_per_class=0.5,
            ),
        )
    )
    fault = [f for f in view.findings if f.subject == "FAULT"]
    report.block("FAULT 를 치명 클래스로 지정했을 때", view.render())

    if fault:
        assert any(f.severity == Severity.CRITICAL.value for f in fault)
        report.note("같은 재현율이라도 FAULT 면 CRITICAL, 다른 클래스면 WARNING 이다.")


def test_정밀도와_재현율은_다른_실패를_말한다() -> None:
    """헛경보가 많은 모델과 놓치는 모델은 다르게 고쳐야 한다."""
    trigger_happy = ConfusionMatrix.from_pairs(
        LABELS,
        [("FAULT", "FAULT")] * 8
        + [("NORMAL", "FAULT")] * 60
        + [("NORMAL", "NORMAL")] * 85
        + [("OVERLOAD", "OVERLOAD")] * 42,
    )
    report.block("헛경보가 많은 모델", trigger_happy.render())

    assert trigger_happy.recall_of("FAULT") == 1.0
    assert trigger_happy.precision_of("FAULT") < 0.2

    findings = EvaluationPolicy().inspect(
        EvaluationResult(split="test", matrix=trigger_happy)
    )
    assert "EVAL_PRECISION_RECALL_IMBALANCE" in codes(findings)
    report.note(
        "트립은 하나도 안 놓쳤다(재현율 100%). "
        "대신 정상 60건을 트립이라고 했다(정밀도 12%). "
        "현장에서는 이 모델을 곧 꺼 버린다."
    )


def test_지연시간도_평가의_일부다(trained) -> None:
    """정확도만 재는 평가는 절반짜리다. 모듈 4 로 이어지는 지점이다."""
    view = trained.evaluation
    report.block(
        "추론 지연시간",
        f"  p50 {view.latency_ms_p50:.3f}ms / p95 {view.latency_ms_p95:.3f}ms",
    )
    assert view.latency_ms_p95 > 0
    report.note(
        "PC 에서 잰 숫자다. 디바이스에서는 10~100배가 된다. "
        "그 사이를 메우는 것이 모듈 4다."
    )


def test_혼동_행렬은_손으로_만들어_확인할_수_있다() -> None:
    matrix = ConfusionMatrix.from_pairs(
        ("A", "B"), [("A", "A"), ("A", "B"), ("B", "B"), ("B", "B")]
    )
    assert matrix.accuracy == 0.75
    assert matrix.recall_of("A") == 0.5
    assert matrix.precision_of("B") == pytest.approx(2 / 3)
    assert matrix.macro_recall == 0.75
