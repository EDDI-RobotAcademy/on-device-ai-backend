"""실습 4-1 — PC에서 돌아간다고 끝난 게 아니다.

    pytest -m lesson_4_1 -s

모듈 3 에서 승인받은 모델이 있다. 정확도도 좋다.
그런데 "이 모델 몇 ms 나와요?"라는 질문에는 아직 답할 수 없다.

**어떻게 재느냐에 따라 답이 몇 배씩 달라지기 때문이다.**
"""

from __future__ import annotations

import pytest

from domain.optimization.benchmark import (
    BenchmarkPolicy,
    BenchmarkResult,
    MeasurementProtocol,
)
from domain.optimization.errors import OptimizationRunNotFound
from domain.shared.errors import IllegalStateTransition
from domain.shared.inspection import Severity
from tests.support import optimization_scenario as os4
from tests.support import report

pytestmark = pytest.mark.lesson_4_1


def codes(findings) -> set[str]:  # noqa: ANN001
    return {f.code for f in findings}


def test_숫자에는_어떻게_쟀는지가_붙어_있어야_한다(optimized) -> None:
    report.section("실습 4-1 · PC에서 돌아간다고 끝난 게 아니다")

    view = optimized.baseline
    report.block("기준 모델 측정", view.render())

    assert view.protocol  # 프로토콜 없이 숫자만 돌려주지 않는다
    assert "warmup=" in view.protocol
    assert "threads=1" in view.protocol
    assert view.p50_ms > 0
    report.note(
        "p50 이 아니라 p95 를 본다. 사이클 타임은 '보통'이 아니라 '최악'으로 지킨다."
    )
    assert view.p95_ms >= view.p50_ms
    assert view.p99_ms >= view.p95_ms


def test_워밍업_없는_측정은_지연시간이_아니다() -> None:
    """첫 호출은 항상 느리다 — 메모리 할당, 커널 선택, 캐시가 비어 있다."""
    naive = BenchmarkResult(
        protocol=MeasurementProtocol(warmup_runs=0, measured_runs=10),
        p50_ms=0.30,
        p95_ms=2.50,
        p99_ms=3.10,
        min_ms=0.28,
        max_ms=3.40,
    )
    findings = BenchmarkPolicy().inspect(naive)
    report.block("워밍업 없이 10번 잰 결과", naive.describe())

    found = codes(findings)
    assert "BENCH_NO_WARMUP" in found
    assert "BENCH_TOO_FEW_RUNS" in found
    assert "BENCH_JITTER_HIGH" in found
    assert any(f.severity is Severity.CRITICAL for f in findings)
    report.note(
        f"p95 가 p50 의 {naive.jitter_ratio:.1f}배. "
        "워밍업을 건너뛰면 첫 호출의 비용이 그대로 분위수에 섞인다."
    )


def test_배치로_잰_것은_지연시간이_아니라_처리량이다() -> None:
    batched = BenchmarkResult(
        protocol=MeasurementProtocol(batch_size=32, threads=8),
        p50_ms=1.0,
        p95_ms=1.1,
        p99_ms=1.2,
        min_ms=0.9,
        max_ms=1.3,
    )
    findings = BenchmarkPolicy().inspect(batched)
    found = codes(findings)

    assert "BENCH_BATCHED" in found
    assert "BENCH_MULTI_THREAD" in found
    report.note(
        "배치 32 · 스레드 8 로 재면 표본당 시간이 짧아 보인다. "
        "현장은 표본이 하나씩 들어오고, 디바이스에는 코어가 하나일 수 있다."
    )


def test_승인받지_않은_모델은_최적화_대상이_아니다(optimization_container, optimized) -> None:
    """순서가 있다. 쓸 수 있는지도 모르는 모델을 먼저 빠르게 만들지 않는다."""
    from application.model.accept_model import (
        AcceptModel,
        AcceptModelCommand,
        ReopenTrainingRun,
        ReopenTrainingRunCommand,
    )

    training_run_id = optimized.training_run_id
    model = optimized.optimization
    # 승인을 되돌린다 — 실습을 위해 일부러.
    ReopenTrainingRun(model.training_runs).execute(
        ReopenTrainingRunCommand(run_id=training_run_id, reason="실습 4-1 확인용")
    )
    try:
        with pytest.raises(IllegalStateTransition) as caught:
            os4.start(
                optimization_container,
                run_id="opt-not-accepted",
                training_run_id=training_run_id,
            )
        report.note(str(caught.value))
        assert "승인" in str(caught.value)
    finally:
        AcceptModel(model.training_runs).execute(
            AcceptModelCommand(run_id=training_run_id)
        )


def test_없는_최적화는_없다고_말한다(optimization_container) -> None:
    with pytest.raises(OptimizationRunNotFound):
        os4.benchmark(optimization_container, "opt-없음")
