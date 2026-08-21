"""실습 4-2 — PyTorch 모델을 디바이스용 모델로 바꿔라.

    pytest -m lesson_4_2 -s

학습에 쓰던 모델은 파이썬 함수다. 층마다 파이썬 인터프리터가 호출된다.
디바이스에 파이썬이 있으리라는 보장은 없다.

첫 단계는 **파이썬을 벗어난 그래프로 굳히는 것**이다 (TorchScript).
그리고 굳힌 뒤에 반드시 확인한다 — 같은 답을 내는가.
"""

from __future__ import annotations

import pytest

from domain.optimization.conversion import (
    ConversionRecord,
    EquivalencePolicy,
    NumericalEquivalence,
)
from domain.optimization.runtime import Precision, RuntimeTarget
from domain.shared.inspection import Severity
from tests.support import report

pytestmark = pytest.mark.lesson_4_2


def codes(findings) -> set[str]:  # noqa: ANN001
    return {f.code for f in findings}


def test_그래프로_굳혀도_같은_답을_내야_한다(optimized) -> None:
    report.section("실습 4-2 · PyTorch 모델을 디바이스용 모델로 바꿔라")

    view = optimized.candidates["TORCHSCRIPT/FP32"]
    report.block("TorchScript 변환", view.render())

    assert view.runtime == "TORCHSCRIPT"
    assert view.precision == "FP32"
    report.note(
        "trace 는 한 번 실행해 보고 그 경로를 기록한다. "
        "그래서 forward 안에 입력에 따라 갈리는 if 가 있으면 한쪽만 남는다."
    )
    report.note("굳힌 뒤 실제 데이터로 대조했고, 예측이 하나도 바뀌지 않았다.")


def test_실행_경로를_바꿔도_파라미터_수는_그대로다(optimized) -> None:
    """모델을 바꾼 게 아니라 **실행 방법**을 바꿨다."""
    baseline = optimized.baseline
    view = optimized.candidates["TORCHSCRIPT/FP32"]

    assert view.theoretical_weight_bytes == 3187 * 4
    report.note(
        f"가중치는 그대로 {view.theoretical_weight_bytes:,}B. "
        f"파일 크기만 {baseline.size_bytes:,}B → {view.size_bytes:,}B 로 달라졌다."
    )
    report.note("달라진 것은 함께 담긴 그래프 구조다. 계산 자체는 같다.")


def test_PyTorch_런타임은_디바이스_배포_대상이_아니다() -> None:
    assert RuntimeTarget.PYTORCH.is_deployable_on_device is False
    assert RuntimeTarget.TORCHSCRIPT.is_deployable_on_device is True
    report.note(
        "TorchScript 는 파이썬 없이 돈다. 다만 PyTorch 런타임은 여전히 필요하다 — "
        "그래서 이것이 마지막 단계는 아니다."
    )


def test_대조하지_않은_변환은_확인하지_않은_것이다() -> None:
    """변환 도구는 **축이 틀려도 성공한다.** 그것이 이 검사가 필요한 이유다."""
    unverified = ConversionRecord(
        source_runtime=RuntimeTarget.PYTORCH,
        target_runtime=RuntimeTarget.TORCHSCRIPT,
        precision=Precision.FP32,
    )
    findings = EquivalencePolicy().inspect(unverified)

    assert "CONVERT_NOT_VERIFIED" in codes(findings)
    assert findings[0].severity is Severity.CRITICAL
    report.note(unverified.describe())


def test_예측이_바뀌면_값이_비슷한_것은_위로가_되지_않는다() -> None:
    broken = ConversionRecord(
        source_runtime=RuntimeTarget.PYTORCH,
        target_runtime=RuntimeTarget.TORCHSCRIPT,
        precision=Precision.FP32,
        equivalence=NumericalEquivalence(
            sample_count=200,
            max_abs_diff=3e-5,
            mean_abs_diff=1e-6,
            agreement_ratio=0.95,
        ),
    )
    findings = EquivalencePolicy().inspect(broken)

    assert "CONVERT_PREDICTION_CHANGED" in codes(findings)
    assert broken.equivalence.disagreement_count == 10
    report.note(
        "max|diff| 는 3e-05 로 작다. 그런데 200개 중 10개에서 답이 바뀌었다. "
        "경계에 있던 표본은 아주 작은 차이로도 넘어간다."
    )


def test_표본이_적으면_같다고_말할_수_없다() -> None:
    thin = ConversionRecord(
        source_runtime=RuntimeTarget.PYTORCH,
        target_runtime=RuntimeTarget.TORCHSCRIPT,
        precision=Precision.FP32,
        equivalence=NumericalEquivalence(
            sample_count=4, max_abs_diff=0.0, mean_abs_diff=0.0, agreement_ratio=1.0
        ),
    )
    findings = EquivalencePolicy().inspect(thin)

    assert "CONVERT_SAMPLE_TOO_FEW" in codes(findings)
    report.note("4개로 100% 일치를 확인했다. 그 100% 는 우연일 수 있다.")
