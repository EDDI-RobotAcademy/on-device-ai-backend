"""실습 4-13 — CPU와 메모리를 실제로 재라.

    pytest -m lesson_4_13 -s

실습 4-5 는 **파일 크기**를 쟀다. 9 KiB 까지 줄였다.
그런데 디바이스에서 프로세스를 죽이는 것은 파일 크기가 아니다.

    모델 파일     16 KiB
    실행 중 RSS  838 MiB

**"모델이 16KB 니까 256MB 보드에 들어갑니다"는 답이 아니다.**

정직하게 밝혀 둘 것: 여기서 재는 RSS 는 **파이썬 프로세스 전체**다.
인터프리터도, numpy 도, torch 도, tensorflow 도 그 안에 있다.
실제 보드는 TFLite 인터프리터만 올린다 — 절대값은 다르다.
그래서 이 실습에서 믿을 수 있는 것은 **차이**다.

    배치를 키웠을 때 얼마나 늘어나는가  → 활성화 메모리
    CPU 시간 ÷ 벽시계 시간              → 코어를 몇 개 썼는가
"""

from __future__ import annotations

import pytest

from application.optimization.measure_resources import MeasureResourcesCommand
from domain.optimization.benchmark import MeasurementProtocol
from domain.optimization.resource import (
    ResourceBudget,
    ResourcePolicy,
    ResourceUsage,
)
from tests.support import report

pytestmark = pytest.mark.lesson_4_13

PROTOCOL = MeasurementProtocol(warmup_runs=15, measured_runs=80)


def _measure(optimized, **kwargs):  # noqa: ANN001, ANN003
    return optimized.optimization.measure_resources().execute(
        MeasureResourcesCommand(run_id=optimized.run_id, **kwargs)
    )


def test_파일_크기와_실행_중_메모리는_다른_축이다(optimized) -> None:
    report.section("실습 4-13 · CPU와 메모리를 실제로 재라")

    views = _measure(optimized, protocol=PROTOCOL)
    report.block(
        "결과물마다",
        "\n".join(
            f"  {v.label:<18}파일 {v.artifact_bytes / 1024:>7.1f} KiB  "
            f"실행 중 {v.peak_rss_bytes / 1024 / 1024:>7.1f} MiB  "
            f"({v.rss_to_artifact_ratio:>7,.0f}배)"
            for v in views
        ),
    )

    assert views
    smallest = min(views, key=lambda v: v.artifact_bytes)
    largest = max(views, key=lambda v: v.artifact_bytes)
    assert largest.artifact_bytes > smallest.artifact_bytes * 1.5
    assert abs(largest.peak_rss_bytes - smallest.peak_rss_bytes) < (
        smallest.peak_rss_bytes * 0.2
    )
    report.note(
        "파일은 INT8 이 FP32 의 절반 이하다. **그런데 실행 중 메모리는 거의 같다.** "
        "대부분이 런타임이기 때문이다 — 여기서 모델을 더 줄여도 소용이 없다."
    )


def test_런타임이_지배하면_경량화가_답이_아니다(optimized) -> None:
    views = _measure(optimized, protocol=PROTOCOL)
    codes = {f.code for v in views for f in v.findings}

    report.block(
        "소견",
        "\n".join(
            f"  {v.label:<18}{f.describe()}" for v in views for f in v.findings
        ),
    )

    assert "RES_RUNTIME_DOMINATES" in codes
    report.note(
        "모델 파일의 5만 배가 실행 중 메모리다. "
        "**여기서 할 일은 모델을 더 줄이는 것이 아니라 런타임을 바꾸는 것이다** — "
        "PyTorch 런타임 대신 TFLite 인터프리터를 쓰면 이 숫자가 통째로 달라진다 (실습 4-4)."
    )


def test_배치를_키우면_늘어나는_것이_활성화_메모리다(optimized) -> None:
    """이 실습에서 유일하게 절대값을 믿을 수 있는 숫자."""
    rows = []
    for batch in (1, 256, 4096):
        view = _measure(
            optimized,
            labels=("TFLITE/FP32",),
            protocol=MeasurementProtocol(
                warmup_runs=10, measured_runs=50, batch_size=batch
            ),
        )[0]
        rows.append((batch, view.model_rss_bytes))

    report.block(
        "배치 → 늘어난 메모리",
        "\n".join(
            f"  배치 {b:>5}  +{n / 1024 / 1024:>7.2f} MiB" for b, n in rows
        ),
    )

    assert rows[-1][1] > rows[0][1]
    report.note(
        "배치를 키우면 **중간 결과를 담을 자리**가 그만큼 더 필요하다. "
        "이것이 활성화 메모리다 — 파라미터와 별개로 잡히고, "
        "임베디드에서는 이쪽이 먼저 터진다."
    )


def test_CPU_사용률이_1을_넘으면_코어를_여러_개_쓴_것이다(optimized) -> None:
    view = _measure(optimized, labels=("TFLITE/FP32",), protocol=PROTOCOL)[0]

    report.block(
        "CPU",
        f"  CPU 시간   {view.cpu_time_ms:>8.1f} ms\n"
        f"  벽시계     {view.wall_time_ms:>8.1f} ms\n"
        f"  사용률     {view.cpu_utilization:>8.2f} 코어",
    )

    assert view.cpu_utilization > 0
    report.note(
        "1.00 이면 코어 하나를 꽉 쓴 것이다. **2.00 이면 두 개를 쓴 것이다** — "
        "코어가 하나뿐인 보드로 가면 그 측정은 절반이 거짓말이 된다 (실습 4-1)."
    )


def test_코어가_하나뿐인_보드_기준으로_판정한다() -> None:
    usage = ResourceUsage(
        label="TFLITE/FP32",
        baseline_rss_bytes=100 * 1024 * 1024,
        peak_rss_bytes=130 * 1024 * 1024,
        cpu_time_ms=3800.0,
        wall_time_ms=1000.0,
        threads=4,
        artifact_bytes=16 * 1024,
    )
    findings = ResourcePolicy(
        budget=ResourceBudget(max_rss_bytes=256 * 1024 * 1024, max_cores=1.0)
    ).inspect(usage)

    report.block("소견", "\n".join(f"  {f.describe()}" for f in findings))
    assert any(f.code == "RES_MULTI_CORE" and f.is_blocking for f in findings)
    report.note(
        "CPU 3.8 코어어치를 썼다. 노트북에서는 빠르게 나왔을 것이다. "
        "**보드에 코어가 하나면 그 숫자는 현장에서 4배가 된다.**"
    )


def test_예산을_넘으면_느려지는_것이_아니라_죽는다() -> None:
    usage = ResourceUsage(
        label="PYTORCH/FP32",
        baseline_rss_bytes=200 * 1024 * 1024,
        peak_rss_bytes=400 * 1024 * 1024,
        cpu_time_ms=900.0,
        wall_time_ms=1000.0,
        artifact_bytes=20 * 1024,
    )
    findings = ResourcePolicy(
        budget=ResourceBudget(max_rss_bytes=256 * 1024 * 1024)
    ).inspect(usage)

    assert any(f.code == "RES_OVER_MEMORY" and f.is_blocking for f in findings)
    report.note(
        "지연시간이 예산을 넘으면 사이클을 놓친다. "
        "**메모리가 예산을 넘으면 프로세스가 사라진다.** "
        "그래서 두 예산은 같은 무게로 다룰 수 없다."
    )
