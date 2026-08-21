"""실습 2-6 — Noise를 제거하지 않으면 AI가 Noise를 배운다.

    pytest -m lesson_2_6 -s

그런데 반대쪽 함정이 더 위험하다.
**과하게 매끈하게 만들면 이상 징후 자체가 지워진다.**
"""

from __future__ import annotations

import pytest

from application.data_quality.measure_noise import MeasureNoiseCommand
from domain.data_quality.noise import FieldNoise, NoiseMeasurement, NoisePolicy
from tests.support import report
from tests.support import quality_scenario as qs

pytestmark = pytest.mark.lesson_2_6


def codes(view) -> set[str]:  # noqa: ANN001
    return {f.code for f in view.findings}


def measure(container, quality_container, name, path):  # noqa: ANN001
    qs.prepare_dataset(container, name, path)
    qs.start(quality_container, f"qa-{name}", name)
    return quality_container.measure_noise().execute(
        MeasureNoiseCommand(assessment_id=f"qa-{name}")
    )


def test_한_채널의_잡음이_전체를_못_쓰게_만든다(
    container, quality_container, quality
) -> None:
    report.section("실습 2-6 · Noise를 제거하지 않으면 AI가 Noise를 배운다")

    view = measure(container, quality_container, "dirty", quality.dirty)
    report.block("잡음 검사", view.render())

    found = codes(view)
    assert "NOISE_SNR_LOW" in found
    assert "NOISE_JITTER" in found
    assert view.verdict == "FAILED"

    snr = next(f for f in view.findings if f.code == "NOISE_SNR_LOW")
    assert snr.subject == "voltage_v"
    report.note(
        f"전압 SNR {snr.measured:.1f}dB (기준 {snr.threshold:.0f}dB). "
        "다섯 채널은 멀쩡한데 전압 하나가 무너졌다."
    )
    report.note(
        f"그래서 이 축의 점수는 {view.score:.1f} 다 — 평균이 아니라 **가장 나쁜 채널**로 낸다. "
        "평균을 내면 1/6 로 희석되어 사라진다."
    )


def test_톱니_패턴은_계통_문제의_신호다(
    container, quality_container, quality
) -> None:
    view = measure(container, quality_container, "dirty", quality.dirty)
    jitter = next(f for f in view.findings if f.code == "NOISE_JITTER")

    assert jitter.measured is not None and jitter.measured > 0.95
    report.note(
        f"부호 반전율 {jitter.measured:.1%}. 백색잡음이면 약 67% 가 나온다. "
        "99% 는 값이 거의 매 표본마다 방향을 바꾼다는 뜻이다."
    )
    report.note(
        "전원 주파수와 샘플링 주파수의 간섭(aliasing)이다. "
        "필터링이 아니라 수집 설정을 고쳐야 한다."
    )


def test_과하게_매끈한_데이터도_막는다() -> None:
    """이동평균을 크게 걸면 SNR 은 좋아지고 사고는 사라진다."""
    result = NoisePolicy().evaluate(
        NoiseMeasurement(
            fields=(
                FieldNoise(
                    field_name="active_power_kw",
                    signal_power=1200.0,
                    noise_power=0.0001,
                    high_frequency_ratio=0.0,  # 고주파가 하나도 없다
                ),
            )
        )
    )
    assert "NOISE_OVERSMOOTHED" in {f.code for f in result.findings}
    report.note(
        "SNR 은 완벽하다. 그런데 짧고 뾰족한 사건이 하나도 없다. "
        "누군가 이미 평활화한 데이터다 — 우리가 찾으려던 것이 그 과정에서 지워졌다."
    )


def test_잡음_측정은_진짜_사건에_끌려가면_안_된다(
    container, quality_container, quality
) -> None:
    """오염 없는 데이터에도 설비 정지(150건)가 있다. 그것은 잡음이 아니다."""
    view = measure(container, quality_container, "clean", quality.clean)
    report.block("오염 없는 데이터의 잡음 검사", view.render())

    assert view.verdict == "PASSED"
    assert view.score == 100.0
    report.note(
        "잔차의 '분산'을 쓰면 정지 구간 150건이 잡음으로 계산되어 SNR 이 무너진다. "
        "그래서 여기서도 MAD 기반 강건 척도를 쓴다 — 실습 2-3 과 같은 이유다."
    )


def test_SNR_은_신호와_잡음의_비율이다() -> None:
    """단위를 손으로 확인해 둔다."""
    clean = FieldNoise(field_name="v", signal_power=100.0, noise_power=1.0)
    dirty = FieldNoise(field_name="v", signal_power=100.0, noise_power=100.0)

    assert clean.snr_db == pytest.approx(20.0)
    assert dirty.snr_db == pytest.approx(0.0)
    report.note("0dB 는 신호와 잡음의 크기가 같다는 뜻이다. 절반은 거짓이다.")
