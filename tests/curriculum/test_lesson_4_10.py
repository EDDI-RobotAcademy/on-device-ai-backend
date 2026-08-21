"""실습 4-10 — 가장 빠른 모델이 아니라 가장 쓸 수 있는 모델을 선택하라.

    pytest -m lesson_4_10 -s

모듈 4 의 게이트다. 모듈 1·2·3 과 같은 자리에 있다.

    모듈 1  이 데이터가 무엇인지 아는가
    모듈 2  이 데이터가 쓸 만한가
    모듈 3  이 모델이 쓸 만한가
    모듈 4  이 모델을 **이 디바이스에** 올릴 수 있는가

선택 규칙은 한 줄이다.
    **예산을 만족하는 후보 중에서 가장 정확한 것.**
"""

from __future__ import annotations

import pytest

from domain.optimization.selection import SelectionObjective
from domain.shared.errors import IllegalStateTransition
from tests.support import optimization_scenario as os4
from tests.support import report

pytestmark = pytest.mark.lesson_4_10


@pytest.fixture
def selectable(optimization_container, optimized):  # noqa: ANN001, ANN201
    """선택을 마음껏 해 볼 수 있는 최적화 하나.

    세션 파이프라인을 그대로 쓰면 첫 선택에서 상태가 굳는다.
    변환은 이미 끝난 결과를 다시 쓰므로 빠르다.
    """
    from domain.optimization.runtime import Precision, RuntimeTarget

    run_id = "opt-selection"
    os4.start(
        optimization_container,
        run_id=run_id,
        training_run_id=optimized.training_run_id,
    )
    os4.benchmark(optimization_container, run_id)
    for runtime, precision in (
        (RuntimeTarget.ONNX, Precision.FP32),
        (RuntimeTarget.TFLITE, Precision.FP32),
        (RuntimeTarget.TFLITE, Precision.INT8),
    ):
        os4.convert(optimization_container, run_id, runtime, precision)
    return run_id


def test_예산_안에서_가장_정확한_것을_고른다(optimization_container, selectable) -> None:
    report.section("실습 4-10 · 가장 빠른 모델이 아니라 가장 쓸 수 있는 모델")

    view = os4.select(optimization_container, selectable)
    report.block("선택 결과", view.render())

    assert view.verdict == "PASSED"
    assert view.has_selection
    report.note(
        f"선택된 것은 {view.selected_label} 다. "
        "'가장 빠른 것'이 아니다 — 예산을 만족한 뒤에는 속도가 아니라 정확도로 고른다."
    )


def test_파이썬을_요구하는_런타임은_후보가_아니다(optimization_container, selectable) -> None:
    view = os4.select(optimization_container, selectable)
    rejected = {label for label, _ in view.rejected}

    report.block(
        "탈락",
        "\n".join(
            f"  ✗ {label}\n" + "\n".join(f"      {r}" for r in reasons)
            for label, reasons in view.rejected
        ),
    )
    assert "PYTORCH/FP32" in rejected
    report.note(
        "기준 모델은 언제나 가장 정확하다. 그런데 디바이스에 파이썬이 없으면 "
        "그 정확도는 쓸 수 없는 정확도다."
    )


def test_예산이_바뀌면_답이_바뀐다(optimization_container, selectable) -> None:
    """이 실습의 핵심. **'최적'은 예산 없이 정의되지 않는다.**"""
    generous = os4.select(optimization_container, selectable)

    # 방금 뽑힌 후보가 **들어가지 못하는** 플래시 용량으로 조인다.
    chosen = next(
        view
        for view in os4.sizes(optimization_container, selectable)
        if view.label == generous.selected_label
    )
    tight_kib = chosen.size_bytes / 1024 - 0.5

    optimization_container.reopen_optimization_run().execute(
        _reopen(selectable, f"저장 예산이 {tight_kib:.1f}KiB 로 줄었다")
    )
    tight = os4.select(
        optimization_container,
        selectable,
        budget=os4.cycle_budget(storage_kib=tight_kib),
    )

    report.block(
        "예산만 바꿨을 때",
        "\n".join(
            [
                f"  저장 64KiB      → 선택 {generous.selected_label} "
                f"({chosen.size_bytes:,}B)",
                f"  저장 {tight_kib:.1f}KiB → 선택 {tight.selected_label}",
            ]
        ),
    )
    assert generous.selected_label != tight.selected_label
    report.note(
        "모델도, 측정값도 하나도 안 바뀌었다. 바뀐 것은 플래시 용량 한 줄이다."
    )
    report.note(
        "그래서 이 판정은 코드가 아니라 **설비가** 정한다. "
        "DeviceBudget 이 요청으로 들어오는 이유다."
    )


def test_예산을_만족하는_후보가_없으면_막는다(optimization_container, selectable) -> None:
    blocked = os4.select(
        optimization_container,
        selectable,
        budget=os4.cycle_budget(latency_p95_ms=0.0001),
    )
    report.block("사이클 타임 0.0001ms", blocked.render())

    assert blocked.verdict == "FAILED"
    assert not blocked.has_selection
    report.note(
        "아무것도 고르지 않는다. **가장 나은 것을 억지로 내놓지 않는다** — "
        "그 순간 예산은 의미를 잃는다."
    )

    with pytest.raises(IllegalStateTransition):
        os4.convert(
            optimization_container,
            selectable,
            _runtime("TORCHSCRIPT"),
            _precision("FP32"),
        )
    report.note("판정된 뒤에는 후보를 몰래 추가할 수 없다. reopen(reason) 을 거쳐야 한다.")


def test_목표를_바꾸면_같은_예산에서도_다른_것을_고른다(optimization_container, selectable) -> None:
    accuracy = os4.select(optimization_container, selectable)
    optimization_container.reopen_optimization_run().execute(
        _reopen(selectable, "플래시가 정말 모자란 상황을 가정한다")
    )
    smallest = os4.select(
        optimization_container, selectable, objective=SelectionObjective.SIZE
    )

    report.block(
        "목표만 바꿨을 때",
        "\n".join(
            [
                f"  ACCURACY → {accuracy.selected_label}",
                f"  SIZE     → {smallest.selected_label}",
            ]
        ),
    )
    assert smallest.selected_label == "TFLITE/INT8"
    report.note(
        "SIZE 목표는 '예산을 만족하는 것 중 가장 작은 것'이다. "
        "예산을 만족한다는 조건이 먼저 붙는다는 점은 같다."
    )


def test_클래스_하나가_무너지면_평균이_좋아도_막힌다() -> None:
    """평균은 소수 클래스의 붕괴를 숨긴다. (실습 4-7 에서 본 그 상황)"""
    from domain.optimization.benchmark import BenchmarkResult, MeasurementProtocol
    from domain.optimization.conversion import ConversionRecord, NumericalEquivalence
    from domain.optimization.identifiers import ArtifactId
    from domain.optimization.runtime import ModelArtifact, Precision, RuntimeTarget
    from domain.optimization.selection import DeviceBudget, SelectionPolicy
    from domain.optimization.tradeoff import (
        ArtifactAccuracy,
        OptimizationCandidate,
        TradeoffTable,
    )

    def candidate(label: str, runtime, precision, accuracy, diff: float) -> OptimizationCandidate:  # noqa: ANN001
        return OptimizationCandidate(
            artifact=ModelArtifact(
                artifact_id=ArtifactId.of(label),
                runtime=runtime,
                precision=precision,
                size_bytes=10_000,
                uri=f"mem://{label}",
            ),
            conversion=ConversionRecord(
                source_runtime=RuntimeTarget.PYTORCH,
                target_runtime=runtime,
                precision=precision,
                equivalence=NumericalEquivalence(
                    sample_count=200,
                    max_abs_diff=diff,
                    mean_abs_diff=diff / 10,
                    agreement_ratio=1.0,
                ),
            ),
            benchmark=BenchmarkResult(
                protocol=MeasurementProtocol(),
                p50_ms=1.0,
                p95_ms=1.2,
                p99_ms=1.3,
                min_ms=0.9,
                max_ms=1.4,
            ),
            accuracy=accuracy,
        )

    baseline = candidate(
        "baseline",
        RuntimeTarget.TFLITE,
        Precision.FP32,
        ArtifactAccuracy(
            accuracy=0.970,
            macro_recall=0.950,
            per_class_recall={"FAULT": 0.90, "OVERLOAD": 0.96, "NORMAL": 0.99},
        ),
        diff=1e-6,  # FP32 변환은 숫자가 거의 그대로여야 한다
    )
    quantized = candidate(
        "quantized",
        RuntimeTarget.TFLITE,
        Precision.INT8,
        ArtifactAccuracy(
            accuracy=0.962,
            macro_recall=0.833,
            per_class_recall={"FAULT": 0.55, "OVERLOAD": 0.96, "NORMAL": 0.99},
        ),
        diff=0.4,  # INT8 은 값이 크게 달라진다 — 그래서 값으로 판정하지 않는다
    )

    certificate = SelectionPolicy(
        budget=DeviceBudget(
            name="설비", latency_p95_ms=30.0, max_class_recall_drop=0.05
        )
    ).evaluate(TradeoffTable(baseline=baseline, candidates=(quantized,)))

    report.block("전체 정확도 −0.8%, FAULT 재현율 −35%", certificate.render())
    verdicts = {v.label: v for v in certificate.verdicts}
    assert not verdicts["TFLITE/INT8"].accepted
    assert any("SELECT_CLASS_RECALL_DROP" in r for r in verdicts["TFLITE/INT8"].reasons)
    assert certificate.selected_label == "TFLITE/FP32"
    report.note(
        "전체 정확도만 봤으면 통과시켰을 후보다. "
        "클래스별로 보면 이 모델은 설비 정지를 절반 가까이 놓친다."
    )


def _reopen(run_id: str, reason: str):  # noqa: ANN202
    from application.optimization.select_model import ReopenOptimizationRunCommand

    return ReopenOptimizationRunCommand(run_id=run_id, reason=reason)


def _runtime(name: str):  # noqa: ANN202
    from domain.optimization.runtime import RuntimeTarget

    return RuntimeTarget(name)


def _precision(name: str):  # noqa: ANN202
    from domain.optimization.runtime import Precision

    return Precision(name)
