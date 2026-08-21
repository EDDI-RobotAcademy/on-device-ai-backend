"""PC에서 돌아간다고 끝난 게 아니다. (실습 4-1)

같은 모델을 같은 기계에서 재도 **어떻게 재느냐에 따라 몇 배씩 달라진다.**

    워밍업 없이 첫 호출을 잰다        → 10배 느리게 나온다
    스레드를 다 쓴다                   → 디바이스에는 코어가 하나일 수 있다
    배치를 32로 잰다                   → 현장은 표본 하나씩 들어온다
    평균만 본다                        → 20번에 한 번 튀는 것이 안 보인다

그래서 '몇 ms 나온다'는 말에는 **어떻게 쟀는지가 함께 붙어야 한다.**
그 '어떻게'가 MeasurementProtocol 이다.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


@dataclass(frozen=True, slots=True)
class MeasurementProtocol:
    """어떻게 잴 것인가. 숫자와 항상 함께 다닌다."""

    warmup_runs: int = 30
    measured_runs: int = 300
    batch_size: int = 1
    threads: int = 1

    def __post_init__(self) -> None:
        if self.warmup_runs < 0:
            raise InvariantViolation("워밍업 횟수는 음수일 수 없다.", subject="warmup_runs")
        if self.measured_runs < 1:
            raise InvariantViolation(
                "측정 횟수는 1 이상이어야 한다.", subject="measured_runs"
            )
        if self.batch_size < 1:
            raise InvariantViolation("배치 크기는 1 이상이어야 한다.", subject="batch_size")
        if self.threads < 1:
            raise InvariantViolation("스레드 수는 1 이상이어야 한다.", subject="threads")

    def describe(self) -> str:
        return (
            f"warmup={self.warmup_runs} runs={self.measured_runs} "
            f"batch={self.batch_size} threads={self.threads}"
        )


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """지연시간 측정 결과.

    평균(mean)이 없다. 일부러 뺐다.
    현장에서 중요한 것은 '보통 얼마나 걸리는가'가 아니라
    **'최악에 얼마나 걸리는가'**다. 사이클 타임은 p95 로 지킨다.
    """

    protocol: MeasurementProtocol
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    activation_bytes: int = 0
    """추론 중에 잡아야 하는 중간 결과의 크기 (분석값).

    임베디드에서는 이 숫자가 모델 파일 크기보다 자주 문제가 된다.
    """

    def __post_init__(self) -> None:
        for name in ("p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"):
            if getattr(self, name) < 0:
                raise InvariantViolation(f"{name} 는 음수일 수 없다.", subject=name)
        if not self.min_ms <= self.p50_ms <= self.p95_ms <= self.p99_ms <= self.max_ms:
            raise InvariantViolation(
                "분위수가 순서대로가 아니다. 측정이 잘못되었다.", subject="percentiles"
            )
        if self.activation_bytes < 0:
            raise InvariantViolation(
                "activation_bytes 는 음수일 수 없다.", subject="activation_bytes"
            )

    @property
    def jitter_ratio(self) -> float:
        """p95 가 p50 의 몇 배인가. 크면 튄다는 뜻이다."""
        if self.p50_ms == 0:
            return 1.0
        return self.p95_ms / self.p50_ms

    @property
    def throughput_per_second(self) -> float:
        if self.p50_ms == 0:
            return 0.0
        return 1000.0 * self.protocol.batch_size / self.p50_ms

    def speedup_over(self, other: BenchmarkResult) -> float:
        """other 대비 몇 배 빠른가. p95 기준이다 — 최악을 기준으로 비교한다."""
        if self.p95_ms == 0:
            return 0.0
        return other.p95_ms / self.p95_ms

    def describe(self) -> str:
        return (
            f"p50 {self.p50_ms:>8.4f}ms  p95 {self.p95_ms:>8.4f}ms  "
            f"p99 {self.p99_ms:>8.4f}ms  지터 {self.jitter_ratio:.2f}배"
        )


@dataclass(frozen=True, slots=True)
class BenchmarkPolicy:
    """측정을 믿어도 되는지에 대한 기준. **숫자보다 먼저 검사한다.**"""

    min_warmup_runs: int = 10
    min_measured_runs: int = 100
    max_jitter_ratio: float = 2.0
    require_single_thread: bool = True
    """디바이스에는 코어가 하나일 수 있다. 그 조건으로 재야 의미가 있다."""

    require_batch_one: bool = True
    """현장은 표본이 하나씩 들어온다. 배치로 재면 throughput 을 latency 로 착각한다."""

    def inspect(self, result: BenchmarkResult) -> tuple[Finding, ...]:
        findings: list[Finding] = []
        protocol = result.protocol

        if protocol.warmup_runs < self.min_warmup_runs:
            findings.append(
                Finding(
                    code="BENCH_NO_WARMUP",
                    message=(
                        f"워밍업 {protocol.warmup_runs}회. "
                        "첫 호출은 항상 느리다 — 메모리 할당, 커널 선택, 캐시가 비어 있다. "
                        "그 숫자를 지연시간이라고 부를 수 없다."
                    ),
                    severity=Severity.CRITICAL,
                    subject="warmup",
                    measured=float(protocol.warmup_runs),
                    threshold=float(self.min_warmup_runs),
                )
            )

        if protocol.measured_runs < self.min_measured_runs:
            findings.append(
                Finding(
                    code="BENCH_TOO_FEW_RUNS",
                    message=(
                        f"{protocol.measured_runs}회로 p95 를 계산했다. "
                        "그 p95 는 다음에 재면 달라진다."
                    ),
                    severity=Severity.WARNING,
                    subject="runs",
                    measured=float(protocol.measured_runs),
                    threshold=float(self.min_measured_runs),
                )
            )

        if self.require_single_thread and protocol.threads > 1:
            findings.append(
                Finding(
                    code="BENCH_MULTI_THREAD",
                    message=(
                        f"스레드 {protocol.threads}개로 쟀다. "
                        "디바이스에 코어가 그만큼 있는지 먼저 확인해야 한다."
                    ),
                    severity=Severity.WARNING,
                    subject="threads",
                    measured=float(protocol.threads),
                    threshold=1.0,
                )
            )

        if self.require_batch_one and protocol.batch_size > 1:
            findings.append(
                Finding(
                    code="BENCH_BATCHED",
                    message=(
                        f"배치 {protocol.batch_size}로 쟀다. "
                        "이건 처리량(throughput)이지 지연시간(latency)이 아니다. "
                        "현장은 표본이 하나씩 들어온다."
                    ),
                    severity=Severity.CRITICAL,
                    subject="batch_size",
                    measured=float(protocol.batch_size),
                    threshold=1.0,
                )
            )

        if result.jitter_ratio > self.max_jitter_ratio:
            findings.append(
                Finding(
                    code="BENCH_JITTER_HIGH",
                    message=(
                        f"p95 가 p50 의 {result.jitter_ratio:.1f}배다. "
                        "가끔 크게 튄다 — 사이클 타임을 지키려면 p50 이 아니라 p95 를 봐야 한다."
                    ),
                    severity=Severity.WARNING,
                    subject="jitter",
                    measured=result.jitter_ratio,
                    threshold=self.max_jitter_ratio,
                )
            )

        return tuple(findings)
