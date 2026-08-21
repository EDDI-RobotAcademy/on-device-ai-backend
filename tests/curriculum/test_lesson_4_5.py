"""실습 4-5 — FP32 모델의 무게를 직접 측정하라.

    pytest -m lesson_4_5 -s

파라미터 3,187개 × 4바이트 = 12,748바이트.
계산은 쉽다. 그런데 파일은 그보다 크다.

이 실습에서 재는 것은 **그 차이**다.
차이를 모르면 다음 실습(FP16/INT8)에서 "왜 이론만큼 안 줄었지?"에 답할 수 없다.
"""

from __future__ import annotations

import pytest

from domain.optimization.identifiers import ArtifactId
from domain.optimization.runtime import ModelArtifact, Precision, RuntimeTarget
from tests.support import report

pytestmark = pytest.mark.lesson_4_5

PARAMETER_COUNT = 3187


def test_이론값은_계산되고_실제값은_측정된다(optimized) -> None:
    report.section("실습 4-5 · FP32 모델의 무게를 직접 측정하라")

    from application.optimization.compare_candidates import InspectArtifactSizesCommand

    views = optimized.optimization.inspect_artifact_sizes().execute(
        InspectArtifactSizesCommand(run_id=optimized.run_id)
    )
    report.block(
        "형식별 크기 분해",
        "\n".join(
            f"  {v.label:<18}{v.size_bytes:>8,}B = 가중치 {v.theoretical_weight_bytes:>7,}"
            f" + 오버헤드 {v.overhead_bytes:>7,}"
            f"  (오버헤드 {v.overhead_bytes / v.size_bytes:.0%})"
            for v in views
        ),
    )

    fp32 = [v for v in views if v.precision == "FP32"]
    assert fp32
    for view in fp32:
        assert view.theoretical_weight_bytes == PARAMETER_COUNT * 4
        assert view.size_bytes > view.theoretical_weight_bytes
    report.note(
        f"가중치는 어느 형식이든 {PARAMETER_COUNT * 4:,}B 로 같다. "
        "달라지는 것은 그래프 구조·연산자 목록·메타데이터다."
    )


def test_작은_모델일수록_오버헤드의_몫이_크다() -> None:
    """이 사실이 다음 두 실습의 결과를 미리 설명한다."""
    small = ModelArtifact(
        artifact_id=ArtifactId.of("small"),
        runtime=RuntimeTarget.TFLITE,
        precision=Precision.FP32,
        size_bytes=16_780,
        uri="mem://small",
        parameter_count=PARAMETER_COUNT,
    )
    big = ModelArtifact(
        artifact_id=ArtifactId.of("big"),
        runtime=RuntimeTarget.TFLITE,
        precision=Precision.FP32,
        size_bytes=4_000_000 + 4_032,
        uri="mem://big",
        parameter_count=1_000_000,
    )
    report.block(
        "같은 오버헤드(4,032B), 다른 모델 크기",
        "\n".join([f"  {small.describe()}", f"  {big.describe()}"]),
    )

    assert small.overhead_ratio > 0.2
    assert big.overhead_ratio < 0.01
    report.note(
        f"파라미터 3,187개 모델에서는 오버헤드가 {small.overhead_ratio:.0%}, "
        f"100만개 모델에서는 {big.overhead_ratio:.1%}."
    )
    report.note(
        "그래서 작은 모델을 INT8 로 줄여도 파일이 1/4 이 되지 않는다. "
        "줄어드는 것은 가중치 부분뿐이기 때문이다."
    )


def test_정밀도가_이론_크기를_결정한다() -> None:
    assert Precision.FP32.bytes_per_weight == 4
    assert Precision.FP16.bytes_per_weight == 2
    assert Precision.INT8.bytes_per_weight == 1
    assert Precision.INT8.is_quantized is True
    assert Precision.FP16.is_quantized is False

    report.block(
        "이론상 가중치 크기",
        "\n".join(
            f"  {p.value:<6}{PARAMETER_COUNT * p.bytes_per_weight:>8,}B"
            for p in Precision
        ),
    )
    report.note("FP16 은 값을 짧게 쓸 뿐 정수로 바꾸지 않는다. INT8 만 양자화다.")


def test_활성값은_파일_크기와_별개로_RAM을_먹는다(optimized) -> None:
    """임베디드에서는 이쪽이 먼저 막히는 일이 잦다."""
    baseline = optimized.baseline
    report.block(
        "메모리 두 종류",
        "\n".join(
            [
                f"  모델 파일 (플래시) : {baseline.size_bytes:>7,}B",
                f"  활성값 추정 (RAM)  : {baseline.activation_bytes:>7,}B",
            ]
        ),
    )
    assert baseline.activation_bytes > 0
    report.note(
        "모델이 64KiB 플래시에 들어가도, 중간 결과가 RAM 을 넘으면 못 돌린다. "
        "실습 4-10 의 예산에 이 항목이 따로 들어가는 이유다."
    )
