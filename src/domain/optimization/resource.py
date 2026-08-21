"""CPU와 메모리를 실제로 재라. (실습 4-13, 4-14)

실습 4-5 는 **파일 크기**를 쟀다. 그런데 디바이스에서 문제가 되는 것은
파일 크기가 아니라 **실행 중에 잡히는 메모리**다.

    파일 42 KiB
    실행 중 190 MiB   ← 런타임, 인터프리터, 활성화, 임시 버퍼

임베디드 보드의 RAM 이 256 MiB 라면 이 차이가 배포 가능 여부를 가른다.
"모델이 42KB 니까 괜찮습니다"는 그래서 답이 아니다.

CPU 도 마찬가지다. 한 코어짜리 보드에 스레드 4개로 잰 숫자를 들고 가면
현장에서 4배 느려진다. **CPU 사용률이 100%를 넘는다는 것은 코어를 여러 개 썼다는 뜻이다.**

그리고 배치. (실습 4-14)

    배치 32로 재면 표본당 시간은 짧다.
    그런데 **첫 표본이 답을 받기까지의 시간은 32배 길다.**

현장의 사이클 타임을 지키는 것은 앞의 숫자가 아니라 뒤의 숫자다.

정직하게 밝혀 둘 것: RSS 는 **프로세스 전체**의 메모리다.
모델만의 것이 아니다. 그래서 절대값보다 **모델을 올리기 전후의 차이**를 본다.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.shared.errors import InvariantViolation
from domain.shared.inspection import Finding, Severity


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """추론을 돌리는 동안 실제로 쓴 자원. Infrastructure 가 잰다."""

    label: str
    baseline_rss_bytes: int
    """모델을 올리기 **전** 프로세스 메모리."""

    peak_rss_bytes: int
    """추론 중 최대 프로세스 메모리."""

    cpu_time_ms: float
    """CPU 가 실제로 일한 시간. 여러 코어를 쓰면 벽시계보다 커진다."""

    wall_time_ms: float
    threads: int = 1
    artifact_bytes: int = 0

    def __post_init__(self) -> None:
        for name in ("baseline_rss_bytes", "peak_rss_bytes", "artifact_bytes"):
            if getattr(self, name) < 0:
                raise InvariantViolation(f"{name} 는 음수일 수 없다.", subject=name)
        for name in ("cpu_time_ms", "wall_time_ms"):
            if getattr(self, name) < 0:
                raise InvariantViolation(f"{name} 는 음수일 수 없다.", subject=name)
        if self.threads < 1:
            raise InvariantViolation("스레드 수는 1 이상이어야 한다.", subject="threads")

    @property
    def model_rss_bytes(self) -> int:
        """모델을 올려서 늘어난 몫. **이것이 모델이 실제로 쓰는 메모리에 가깝다.**"""
        return max(0, self.peak_rss_bytes - self.baseline_rss_bytes)

    @property
    def cpu_utilization(self) -> float:
        """CPU 시간 ÷ 벽시계 시간. 1.0 을 넘으면 코어를 여러 개 쓴 것이다."""
        if self.wall_time_ms <= 0:
            return 0.0
        return self.cpu_time_ms / self.wall_time_ms

    @property
    def rss_to_artifact_ratio(self) -> float:
        """실행 중 메모리가 파일 크기의 몇 배인가."""
        if self.artifact_bytes == 0:
            return 0.0
        return self.peak_rss_bytes / self.artifact_bytes

    def describe(self) -> str:
        return (
            f"{self.label:<20}"
            f"RSS {self.peak_rss_bytes / 1024 / 1024:>8.1f} MiB "
            f"(모델 몫 {self.model_rss_bytes / 1024 / 1024:>6.1f})  "
            f"CPU {self.cpu_utilization:>5.2f} 코어  "
            f"파일 {self.artifact_bytes / 1024:>7.1f} KiB"
        )


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    """디바이스가 내줄 수 있는 것. (실습 4-13)

    실습 4-10 의 DeviceBudget 이 지연시간과 크기를 봤다면,
    여기서는 **실행 중 자원**을 본다.
    """

    max_rss_bytes: int = 256 * 1024 * 1024
    max_cores: float = 1.0
    """보드에 쓸 수 있는 코어 수. 대개 1이다."""

    def describe(self) -> str:
        return (
            f"RAM {self.max_rss_bytes / 1024 / 1024:.0f} MiB / "
            f"코어 {self.max_cores:g}개"
        )


@dataclass(frozen=True, slots=True)
class ResourcePolicy:
    """이 자원 사용을 받아들일 수 있는가. (실습 4-13)"""

    budget: ResourceBudget = ResourceBudget()
    warn_rss_ratio: float = 0.8
    """예산의 이만큼을 넘으면 여유가 없다는 뜻이다."""

    max_rss_to_artifact_ratio: float = 100.0

    def inspect(self, usage: ResourceUsage) -> tuple[Finding, ...]:
        findings: list[Finding] = []

        if usage.peak_rss_bytes > self.budget.max_rss_bytes:
            findings.append(
                Finding(
                    code="RES_OVER_MEMORY",
                    message=(
                        f"이 프로세스가 쓰는 메모리가 "
                        f"{usage.peak_rss_bytes / 1024 / 1024:.0f} MiB 다. "
                        f"보드가 내주는 것은 "
                        f"{self.budget.max_rss_bytes / 1024 / 1024:.0f} MiB — "
                        "**넘으면 느려지는 것이 아니라 OOM 으로 죽는다.** "
                        "다만 이 숫자에는 인터프리터와 라이브러리가 함께 들어 있다. "
                        "보드에 올릴 때 무엇이 같이 올라가는지 먼저 따져야 한다."
                    ),
                    severity=Severity.CRITICAL,
                    subject=usage.label,
                    measured=float(usage.peak_rss_bytes),
                    threshold=float(self.budget.max_rss_bytes),
                )
            )
        elif usage.peak_rss_bytes > self.budget.max_rss_bytes * self.warn_rss_ratio:
            findings.append(
                Finding(
                    code="RES_MEMORY_TIGHT",
                    message=(
                        f"예산의 "
                        f"{usage.peak_rss_bytes / self.budget.max_rss_bytes:.0%} 를 쓴다. "
                        "**현장에는 이 프로세스만 도는 것이 아니다** — "
                        "로그 수집기도, OTA 에이전트도 같은 RAM 을 쓴다."
                    ),
                    severity=Severity.WARNING,
                    subject=usage.label,
                    measured=float(usage.peak_rss_bytes),
                    threshold=float(self.budget.max_rss_bytes),
                )
            )

        if usage.cpu_utilization > self.budget.max_cores + 0.2:
            findings.append(
                Finding(
                    code="RES_MULTI_CORE",
                    message=(
                        f"CPU 를 {usage.cpu_utilization:.2f} 코어어치 썼다. "
                        f"보드에는 {self.budget.max_cores:g}개뿐이다 — "
                        "**여기서 잰 지연시간은 현장에서 그대로 나오지 않는다.** "
                        "스레드를 1로 고정하고 다시 재야 한다 (실습 4-1)."
                    ),
                    severity=Severity.CRITICAL,
                    subject=usage.label,
                    measured=usage.cpu_utilization,
                    threshold=self.budget.max_cores,
                )
            )

        if (
            usage.artifact_bytes
            and usage.rss_to_artifact_ratio > self.max_rss_to_artifact_ratio
        ):
            findings.append(
                Finding(
                    code="RES_RUNTIME_DOMINATES",
                    message=(
                        f"실행 중 메모리가 모델 파일의 "
                        f"{usage.rss_to_artifact_ratio:.0f}배다. "
                        "**모델을 더 줄여도 이 숫자는 거의 안 움직인다** — "
                        "대부분이 런타임과 인터프리터다. "
                        "여기서는 모델 경량화가 아니라 런타임 선택이 답이다 (실습 4-3, 4-4)."
                    ),
                    severity=Severity.WARNING,
                    subject=usage.label,
                    measured=usage.rss_to_artifact_ratio,
                    threshold=self.max_rss_to_artifact_ratio,
                )
            )

        return tuple(findings)


# ---------------------------------------------------------------------------
# 배치 (실습 4-14)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BatchPoint:
    """배치 크기 하나에서 잰 값."""

    batch_size: int
    p50_ms: float
    p95_ms: float

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise InvariantViolation("배치 크기는 1 이상이어야 한다.", subject="batch_size")
        if self.p50_ms < 0 or self.p95_ms < 0:
            raise InvariantViolation("지연시간은 음수일 수 없다.", subject="latency")

    @property
    def per_sample_ms(self) -> float:
        """표본 하나당 평균 시간. **처리량 관점의 숫자다.**"""
        return self.p50_ms / self.batch_size

    @property
    def first_answer_ms(self) -> float:
        """첫 표본이 답을 받기까지의 시간.

        배치는 다 모여야 계산이 시작된다. 그래서 배치 전체 시간이 곧 이 시간이다.
        **현장의 사이클 타임을 지키는 것은 이 숫자다.**
        """
        return self.p50_ms

    @property
    def throughput_per_second(self) -> float:
        if self.p50_ms == 0:
            return 0.0
        return 1000.0 * self.batch_size / self.p50_ms


@dataclass(frozen=True, slots=True)
class BatchScaling:
    """배치 크기를 바꿔 가며 잰 결과. (실습 4-14)"""

    points: tuple[BatchPoint, ...]

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise InvariantViolation(
                "배치 비교는 최소 두 지점이 필요하다.", subject="points"
            )
        if self.points[0].batch_size != 1:
            raise InvariantViolation(
                "첫 지점은 배치 1이어야 한다. **기준은 표본 하나다.**",
                subject="points",
            )

    @property
    def single(self) -> BatchPoint:
        return self.points[0]

    @property
    def largest(self) -> BatchPoint:
        return max(self.points, key=lambda p: p.batch_size)

    @property
    def throughput_gain(self) -> float:
        """배치로 얻는 처리량 배수."""
        base = self.single.throughput_per_second
        return self.largest.throughput_per_second / base if base else 0.0

    @property
    def latency_cost(self) -> float:
        """그 대가로 늘어난 '첫 답까지의 시간' 배수."""
        base = self.single.first_answer_ms
        return self.largest.first_answer_ms / base if base else 0.0

    def render(self) -> str:
        header = (
            f"{'배치':>6}{'전체 p50':>12}{'표본당':>10}"
            f"{'첫 답까지':>12}{'처리량/s':>12}"
        )
        lines = [header, "-" * len(header)]
        for point in self.points:
            lines.append(
                f"{point.batch_size:>6}{point.p50_ms:>11.3f}ms"
                f"{point.per_sample_ms:>9.3f}ms{point.first_answer_ms:>11.3f}ms"
                f"{point.throughput_per_second:>12,.0f}"
            )
        lines.append("-" * len(header))
        lines.append(
            f"  배치를 키우면 처리량 {self.throughput_gain:.1f}배, "
            f"첫 답까지 {self.latency_cost:.1f}배"
        )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class CycleTimePolicy:
    """사이클 타임을 지킬 수 있는가. (실습 4-14)"""

    cycle_time_ms: float
    """현장이 요구하는 응답 시간. 라인 속도가 정한다."""

    def __post_init__(self) -> None:
        if self.cycle_time_ms <= 0:
            raise InvariantViolation(
                "사이클 타임은 0보다 커야 한다.", subject="cycle_time_ms"
            )

    def inspect(self, scaling: BatchScaling) -> tuple[Finding, ...]:
        findings: list[Finding] = []

        for point in scaling.points:
            if point.batch_size == 1:
                continue
            if point.first_answer_ms > self.cycle_time_ms:
                findings.append(
                    Finding(
                        code="BATCH_MISSES_CYCLE_TIME",
                        message=(
                            f"배치 {point.batch_size}에서 첫 답까지 "
                            f"{point.first_answer_ms:.2f}ms 걸린다. "
                            f"사이클 타임은 {self.cycle_time_ms:g}ms 다 — "
                            "**표본당 시간이 짧아진 것과 제때 답하는 것은 다른 이야기다.**"
                        ),
                        severity=Severity.CRITICAL,
                        subject=str(point.batch_size),
                        measured=point.first_answer_ms,
                        threshold=self.cycle_time_ms,
                    )
                )

        if scaling.throughput_gain > 1.5:
            findings.append(
                Finding(
                    code="BATCH_THROUGHPUT_GAIN",
                    message=(
                        f"배치를 키우면 처리량이 {scaling.throughput_gain:.1f}배가 된다. "
                        "**여러 대를 한 서버가 맡는 구조라면 의미가 있다** — "
                        "디바이스 한 대에서 한 장씩 보는 구조에서는 의미가 없다."
                    ),
                    severity=Severity.INFO,
                    subject=str(scaling.largest.batch_size),
                    measured=scaling.throughput_gain,
                )
            )

        return tuple(findings)
