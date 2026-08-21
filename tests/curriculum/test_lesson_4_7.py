"""실습 4-7 — INT8로 줄이고 정확도가 얼마나 살아남는지 확인하라.

    pytest -m lesson_4_7 -s

INT8 은 앞의 둘과 다르다. 실수를 **정수 눈금 위에 올린다.**

    실수 범위 [-3.2, 4.8] 을 256칸으로 나눠 -128 ~ 127 에 대응시킨다

그러니 값은 반드시 달라진다. 여기서 물어야 할 질문이 바뀐다.

    "숫자가 같은가?"  →  아니다. 물을 필요도 없다.
    "**답이 같은가?**"  →  이것만 묻는다.
"""

from __future__ import annotations

import pytest

from domain.optimization.tradeoff import ArtifactAccuracy
from tests.support import report

pytestmark = pytest.mark.lesson_4_7


def test_값은_크게_달라지는데_답은_그대로일_수_있다(optimized) -> None:
    report.section("실습 4-7 · INT8로 줄이고 정확도가 얼마나 살아남는지 확인하라")

    fp32 = optimized.candidates["TFLITE/FP32"]
    int8 = optimized.candidates["TFLITE/INT8"]

    report.block(
        "정밀도별 동등성",
        "\n".join(
            [
                f"  FP32 : {fp32.equivalence}",
                f"  FP16 : {optimized.candidates['TFLITE/FP16'].equivalence}",
                f"  INT8 : {int8.equivalence}",
            ]
        ),
    )
    report.note(
        "INT8 의 max|diff| 는 0.1 을 넘는다. FP32 기준(1e-4)으로 재면 즉시 탈락이다."
    )
    report.note(
        "그런데 argmax 는 100% 일치한다. **분류 모델이 내는 답은 크기 순서지 값이 아니다.**"
    )
    assert "argmax 일치 100.00%" in int8.equivalence
    assert int8.findings == ()  # 정밀도에 맞는 기준으로 재면 통과한다


def test_대표_데이터_없이는_정수_눈금을_정할_수_없다(optimization_container, optimized) -> None:
    """INT8 변환에는 **값의 범위**가 필요하다. 그 범위는 데이터에서 온다."""
    from domain.optimization.errors import ConversionFailed
    from domain.optimization.runtime import Precision, RuntimeTarget
    from infrastructure.optimization.exporters import TFLiteExporter
    from tests.support import optimization_scenario as os4

    os4.start(
        optimization_container,
        run_id="opt-int8-norep",
        training_run_id=optimized.training_run_id,
    )
    os4.benchmark(optimization_container, "opt-int8-norep")

    starved = TFLiteExporter(
        optimization_container.registry,
        optimization_container.runtimes,
        optimization_container.artifact_dir,
        representative_samples=0,
    )
    run = optimization_container.runs.find_by_id(_id("opt-int8-norep"))

    with pytest.raises(ConversionFailed) as caught:
        starved.export(run.baseline, RuntimeTarget.TFLITE, Precision.INT8)
    report.note(str(caught.value))
    report.note(
        "그리고 그 대표 데이터는 **train 분할**에서 뽑는다. "
        "test 에서 뽑으면 평가 데이터가 모델 안으로 새어 들어간다 (실습 1-7)."
    )


def test_전체_정확도는_한_클래스의_붕괴를_숨긴다() -> None:
    """이 실습의 진짜 위험. INT8 은 대개 소수 클래스부터 무너뜨린다."""
    baseline = ArtifactAccuracy(
        accuracy=0.970,
        macro_recall=0.950,
        per_class_recall={"FAULT": 0.90, "OVERLOAD": 0.96, "NORMAL": 0.99},
    )
    quantized = ArtifactAccuracy(
        accuracy=0.962,
        macro_recall=0.833,
        per_class_recall={"FAULT": 0.55, "OVERLOAD": 0.96, "NORMAL": 0.99},
    )

    report.block(
        "양자화 전후 클래스별 재현율",
        "\n".join(
            [
                f"  {'클래스':<10}{'FP32':>8}{'INT8':>8}{'차이':>8}",
                *[
                    f"  {label:<10}{baseline.per_class_recall[label]:>8.2f}"
                    f"{quantized.per_class_recall[label]:>8.2f}"
                    f"{quantized.per_class_recall[label] - baseline.per_class_recall[label]:>+8.2f}"
                    for label in baseline.per_class_recall
                ],
            ]
        ),
    )

    assert quantized.drop_from(baseline) == pytest.approx(0.008)
    label, drop = quantized.worst_class_drop_from(baseline)
    assert label == "FAULT"
    assert drop == pytest.approx(0.35)

    report.note(
        f"전체 정확도는 {quantized.drop_from(baseline):.1%} 밖에 안 떨어졌다. "
        "보고서에는 '정확도 유지'라고 쓸 수 있는 숫자다."
    )
    report.note(
        f"그런데 FAULT 재현율은 {drop:.0%} 떨어졌다. "
        "설비 정지를 절반 가까이 놓치기 시작했다는 뜻이다."
    )
    report.note(
        "그래서 DeviceBudget 에 max_class_recall_drop 이 따로 있다 (실습 4-10). "
        "**평균만 보면 이것을 놓친다.**"
    )


def test_이_모델에서는_INT8이_정확도를_깎지_않았다(optimized) -> None:
    """정직하게 기록한다. 항상 떨어지는 것은 아니다."""
    fp32 = optimized.baseline
    int8 = optimized.candidates["TFLITE/INT8"]

    assert int8.macro_recall == pytest.approx(1.0)
    report.note(
        "이 모델·이 데이터에서는 INT8 이 정확도를 전혀 깎지 않았다. "
        "층이 얕고 값의 분포가 좁아서 256칸으로도 충분했다."
    )
    report.note(
        f"대신 크기는 {fp32.size_bytes:,}B → {int8.size_bytes:,}B 로 줄었고, "
        "이 표에서 실제로 교환된 것은 정확도가 아니라 **속도**였다 (실습 4-8)."
    )


def _id(run_id: str):  # noqa: ANN202
    from domain.optimization.identifiers import OptimizationRunId

    return OptimizationRunId.of(run_id)
