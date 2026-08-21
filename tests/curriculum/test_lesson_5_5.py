"""실습 5-5 — Latency가 무너지면 무엇이 달라졌는가?

    pytest -m lesson_5_5 -s

모델은 안 바뀌었다. 바이트 하나 안 바뀌었다.
그런데 현장 p95 가 벤치마크의 20배가 됐다.

**바뀐 것은 환경이다.** 그리고 그 사실을 아는 것이 조치의 출발점이다.
"""

from __future__ import annotations

import pytest

from domain.operations.latency import LatencyPolicy, LatencyProfile
from domain.operations.window import ObservationWindow
from domain.shared.inspection import Severity
from infrastructure.monitoring.inference_log_store import slice_windows
from tests.support import operations_scenario as os5
from tests.support import report

pytestmark = pytest.mark.lesson_5_5


def codes(findings) -> set[str]:  # noqa: ANN001
    return {f.code for f in findings}


def window(label: str = "w", samples: int = 3000) -> ObservationWindow:
    return ObservationWindow(
        label=label,
        started_at="2026-05-23 00:00:00",
        ended_at="2026-05-23 07:59:59",
        sample_count=samples,
    )


def test_현장은_벤치마크보다_느리다(deployed) -> None:
    report.section("실습 5-5 · Latency가 무너지면 무엇이 달라졌는가?")

    first, last = deployed.reports[0], deployed.reports[-1]
    report.block(
        "1일차 vs 4일차",
        "\n".join(
            [
                f"  벤치마크(모듈 4) p95 : 0.0031ms",
                f"  1일차 현장       p95 : {first.p95_ms:.4f}ms",
                f"  4일차 현장       p95 : {last.p95_ms:.4f}ms",
            ]
        ),
    )
    assert first.p95_ms > 0.003
    report.note(
        "1일차부터 이미 10배 가까이 느리다. **이건 정상이다** — "
        "PC 와 임베디드 CPU 는 원래 다르다. 그래서 예산에 여유를 둔다."
    )
    assert last.p95_ms > first.p95_ms * 2
    report.note("4일차에는 거기서 또 2.5배가 됐다. 이쪽이 사건이다.")


def test_전부가_아니라_한_대다(deployed, operations_container) -> None:
    """**평균은 이것을 숨긴다.**"""
    from domain.operations.identifiers import DeploymentId

    records = operations_container.logs.all_records(
        DeploymentId.of(deployed.deployment_id)
    )
    rows = []
    for device in ("DEV-01", "DEV-02", "DEV-03"):
        windows = slice_windows(records, hours=8, device_id=device)
        view = os5.observe(
            operations_container, windows[-1], open_incident=False
        )
        rows.append((device, view.p95_ms))

    report.block(
        "4일차 마지막 창 — 디바이스별 p95",
        "\n".join(f"  {device:<10}{p95:>10.4f}ms" for device, p95 in rows),
    )
    slowest = max(rows, key=lambda r: r[1])
    fastest = min(rows, key=lambda r: r[1])
    assert slowest[1] > fastest[1] * 2
    report.note(
        f"{slowest[0]} 만 {slowest[1] / fastest[1]:.1f}배 느리다. 나머지 두 대는 멀쩡하다."
    )
    report.note(
        "모델 문제였다면 세 대가 같이 느려진다. 한 대만 느리면 **그 디바이스의 문제**다 — "
        "팬 고장, 발열, 다른 프로세스, 전원 정책."
    )
    report.note(
        "디바이스를 로그에 안 남겼으면(실습 5-3) 이 구분을 할 수 없다."
    )


def test_모델을_의심하기_전에_환경을_의심한다() -> None:
    profile = LatencyProfile(
        window=window(),
        p50_ms=0.025,
        p95_ms=0.075,
        p99_ms=0.120,
        max_ms=0.400,
    )
    findings = LatencyPolicy(cycle_budget_ms=30.0, max_regression_ratio=12.0).inspect(
        profile, baseline_p95_ms=0.0031
    )
    report.block("벤치마크 0.0031ms → 현장 0.075ms", "\n".join(
        f"  - {f.describe()}" for f in findings
    ))

    assert "OPS_LATENCY_REGRESSION" in codes(findings)
    message = next(f for f in findings if f.code == "OPS_LATENCY_REGRESSION").message
    assert "모델은 안 바뀌었다" in message
    report.note(
        "의심할 곳은 넷이다 — 발열로 클럭이 내려갔거나, 다른 프로세스가 코어를 가져갔거나, "
        "큐가 밀렸거나, 전원이 절전으로 들어갔다."
    )


def test_사이클_타임을_넘으면_그때부터_CRITICAL_이다() -> None:
    """느려진 것과 못 지키는 것은 다르다."""
    slow = LatencyProfile(
        window=window(), p50_ms=20.0, p95_ms=45.0, p99_ms=60.0, max_ms=90.0
    )
    findings = LatencyPolicy(cycle_budget_ms=30.0).inspect(slow, baseline_p95_ms=0.0031)

    assert "OPS_OVER_CYCLE_BUDGET" in codes(findings)
    assert any(f.severity is Severity.CRITICAL for f in findings)
    report.note(
        "20배 느려져도 사이클 안에 들어오면 WARNING 이다. "
        "사이클을 넘는 순간 CRITICAL 이 된다 — **설비가 부품을 기다리기 시작한다.**"
    )


def test_끝나지_않은_추론은_분위수에_안_잡힌다() -> None:
    """p95 만 보면 안 보이는 실패."""
    timeouts = LatencyProfile(
        window=window(samples=1000),
        p50_ms=2.0,
        p95_ms=4.0,
        p99_ms=5.0,
        max_ms=6.0,
        timeout_count=12,
    )
    findings = LatencyPolicy(cycle_budget_ms=30.0).inspect(
        timeouts, baseline_p95_ms=2.0
    )
    report.block("p95 4ms, 그런데 12건은 아예 안 끝났다", "\n".join(
        f"  - {f.describe()}" for f in findings
    ))

    assert "OPS_TIMEOUT" in codes(findings)
    assert timeouts.timeout_ratio > 0
    report.note(
        "분위수는 **끝난 것들만** 가지고 계산한다. "
        "안 끝난 12건은 어느 분위수에도 안 들어간다 — 따로 세지 않으면 없는 일이 된다."
    )


def test_평소엔_괜찮은데_가끔_튄다() -> None:
    jittery = LatencyProfile(
        window=window(), p50_ms=2.0, p95_ms=9.0, p99_ms=20.0, max_ms=45.0
    )
    findings = LatencyPolicy(cycle_budget_ms=30.0, max_jitter_ratio=3.0).inspect(
        jittery, baseline_p95_ms=2.0
    )
    assert "OPS_LATENCY_JITTER" in codes(findings)
    report.note(
        f"p95 가 p50 의 {jittery.jitter_ratio:.1f}배다. "
        "평균은 2ms 라고 말하지만, 20번에 한 번은 9ms 다."
    )
    report.note("대개 다른 작업과 겹치는 순간이다. 그 작업을 찾는 것이 조치다.")
