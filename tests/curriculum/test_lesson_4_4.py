"""실습 4-4 — TFLite로 모델을 디바이스에 맞춰라.

    pytest -m lesson_4_4 -s

여기서 프레임워크가 통째로 바뀐다. PyTorch → TensorFlow.

가능한 이유는 하나다.
**모듈 3 에서 `ModelArchitecture` 를 nn.Module 이 아니라 '명세'로 만들어 두었기 때문이다.**

같은 명세로 Keras 모델을 조립하고 가중치만 옮기면 된다.
단, 축 순서를 바꿔야 한다 — 그리고 축을 틀려도 변환은 **성공한다.**
"""

from __future__ import annotations

import numpy as np
import pytest

from infrastructure.errors import UnsupportedSourceFormat
from tests.support import report

pytestmark = pytest.mark.lesson_4_4


def test_프레임워크를_바꿔도_같은_답이_나온다(optimized) -> None:
    report.section("실습 4-4 · TFLite로 모델을 디바이스에 맞춰라")

    view = optimized.candidates["TFLITE/FP32"]
    report.block("TFLite 변환 (PyTorch → Keras → TFLite)", view.render())

    assert view.runtime == "TFLITE"
    assert "argmax 일치 100.00%" in view.equivalence
    report.note(
        "PyTorch 로 학습한 가중치가 TensorFlow 인터프리터에서 같은 답을 냈다. "
        "명세가 있으면 프레임워크는 교체 가능한 부품이 된다 (CLAUDE.md §14)."
    )


def test_축을_틀리면_조용히_다른_모델이_된다(optimized) -> None:
    """이 실습의 핵심. **변환기는 축이 틀려도 성공한다.**"""
    from domain.model.identifiers import ModelVersionId
    from infrastructure.optimization.keras_bridge import build_keras_model

    optimization = optimized.optimization
    run = _run(optimization, optimized.run_id)
    trained = optimization.registry.get(
        ModelVersionId.of(run.baseline.model_version_id)
    )
    state = dict(trained.module.state_dict())

    correct = build_keras_model(trained.architecture, state)
    # 마지막 Dense 의 가중치만 전치하지 않고 넣어 본다 — 정사각이라 모양이 맞는다.
    sabotaged = build_keras_model(trained.architecture, state)
    dense = sabotaged.layers[-1]
    weight, bias = dense.get_weights()
    dense.set_weights([weight[:, ::-1], bias])  # 출력 클래스 순서를 뒤집는다

    sample = trained.dataset.features["test"][:8].astype("float32")
    good = np.asarray(correct(sample, training=False))
    bad = np.asarray(sabotaged(sample, training=False))

    report.block(
        "축을 뒤집은 모델",
        "\n".join(
            [
                f"  올바른 변환의 예측 : {good.argmax(axis=1).tolist()}",
                f"  뒤집힌 변환의 예측 : {bad.argmax(axis=1).tolist()}",
            ]
        ),
    )
    assert good.shape == bad.shape  # 모양은 똑같다. 오류도 나지 않는다.
    assert not np.array_equal(good.argmax(axis=1), bad.argmax(axis=1))
    report.note(
        "모양이 같으니 아무도 못 알아챈다. 예외도 안 난다. "
        "**대조하지 않으면 이 모델이 현장에 나간다.**"
    )


def test_지원하지_않는_구조는_지원하지_않는다고_말한다(optimized) -> None:
    """조용히 비슷한 것을 만들어 주는 것이 가장 나쁜 실패다."""
    from domain.model.architecture import ArchitectureKind, ModelArchitecture
    from domain.model.identifiers import ModelVersionId
    from domain.model.tensor_spec import TensorLayout, TensorSpec
    from infrastructure.optimization.keras_bridge import build_keras_model

    optimization = optimized.optimization
    run = _run(optimization, optimized.run_id)
    trained = optimization.registry.get(
        ModelVersionId.of(run.baseline.model_version_id)
    )
    cnn2d = ModelArchitecture(
        kind=ArchitectureKind.CNN2D,
        input_spec=TensorSpec(shape=(64, 64, 1), layout=TensorLayout.CHANNEL_LAST),
        class_count=2,
        hidden_channels=(16, 32),
    )

    with pytest.raises(UnsupportedSourceFormat) as caught:
        build_keras_model(cnn2d, trained.module.state_dict())
    report.note(str(caught.value))
    report.note(
        "이미지 모델(실습 3-3)의 TFLite 경로는 아직 없다. "
        "'아직 없다'고 말하는 것과 조용히 이상한 것을 내주는 것은 완전히 다르다."
    )


def test_변환_실패도_기록으로_남는다(optimization_container, optimized) -> None:
    """'TFLite 로는 안 되더라'가 구전으로만 남으면 다음 사람이 또 해 본다."""
    from domain.optimization.errors import ConversionFailed
    from domain.optimization.runtime import Precision, RuntimeTarget
    from tests.support import optimization_scenario as os4

    os4.start(
        optimization_container,
        run_id="opt-rejection",
        training_run_id=optimized.training_run_id,
    )
    os4.benchmark(optimization_container, "opt-rejection")

    # 지원하지 않는 구조로 바꿔치기해 실패를 만든다.
    original = optimization_container.exporter

    class _AlwaysFails:
        def supports(self, runtime, precision) -> bool:  # noqa: ANN001
            return True

        def export(self, baseline, runtime, precision):  # noqa: ANN001, ANN201
            raise ConversionFailed(
                "이 연산자는 TFLite 에 없다: custom_op_42", subject="TFLITE"
            )

    optimization_container.exporter = _AlwaysFails()
    try:
        with pytest.raises(ConversionFailed):
            os4.convert(
                optimization_container,
                "opt-rejection",
                RuntimeTarget.TFLITE,
                Precision.INT8,
            )
    finally:
        optimization_container.exporter = original

    view = optimization_container.get_optimization_run().execute(
        _query("opt-rejection")
    )
    report.block(
        "남은 실패 기록",
        "\n".join(f"  ✗ {label}: {reason}" for label, reason in view.rejections),
    )
    assert view.rejections
    assert "custom_op_42" in view.rejections[0][1]


def _run(optimization, run_id: str):  # noqa: ANN001, ANN202
    from domain.optimization.identifiers import OptimizationRunId

    return optimization.runs.find_by_id(OptimizationRunId.of(run_id))


def _query(run_id: str):  # noqa: ANN202
    from application.optimization.get_optimization_run import GetOptimizationRunQuery

    return GetOptimizationRunQuery(run_id=run_id)
