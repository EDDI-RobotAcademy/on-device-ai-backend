"""자원 사용 실측 어댑터. (실습 4-13, 4-14)

`psutil` 로 **프로세스 전체**의 메모리와 CPU 시간을 잰다.

정직하게 밝혀 둘 것이 둘 있다.

    1. RSS 는 프로세스 전체다. 모델만의 것이 아니다.
       그래서 모델을 올리기 **전후의 차이**를 함께 돌려준다.

    2. 파이썬 프로세스에는 인터프리터·numpy·torch 가 이미 올라와 있다.
       실제 임베디드 보드의 숫자와 절대값은 다르다.
       **비교는 절대값이 아니라 결과물 사이의 차이로 해야 한다.**

이 한계를 숨기고 "190MiB 나옵니다"라고 쓰면 그 숫자로 보드를 고르게 된다.
"""

from __future__ import annotations

import gc
import os
import time

import numpy as np
import psutil

from domain.optimization.benchmark import MeasurementProtocol
from domain.optimization.resource import BatchPoint, BatchScaling, ResourceUsage
from domain.optimization.runtime import ModelArtifact
from infrastructure.optimization.runtime_registry import RuntimeRegistry


class ProcessResourceMeter:
    """domain.optimization.ports.ResourceMeter 구현. (실습 4-13)"""

    def __init__(self, runtimes: RuntimeRegistry) -> None:
        self._runtimes = runtimes
        self._process = psutil.Process(os.getpid())

    def measure(
        self, artifact: ModelArtifact, protocol: MeasurementProtocol
    ) -> ResourceUsage:
        loaded = self._runtimes.require(artifact.artifact_id)

        gc.collect()
        baseline_rss = self._process.memory_info().rss

        shape = (protocol.batch_size, *loaded.input_shape)
        sample = np.zeros(shape, dtype="float32")
        for _ in range(protocol.warmup_runs):
            loaded.predict(sample)

        peak_rss = max(baseline_rss, self._process.memory_info().rss)
        cpu_before = self._process.cpu_times()
        wall_before = time.perf_counter()

        for index in range(protocol.measured_runs):
            loaded.predict(sample)
            if index % 25 == 0:  # 매번 재면 측정이 측정을 방해한다
                peak_rss = max(peak_rss, self._process.memory_info().rss)

        wall_ms = (time.perf_counter() - wall_before) * 1000.0
        cpu_after = self._process.cpu_times()
        cpu_ms = (
            (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
        ) * 1000.0
        peak_rss = max(peak_rss, self._process.memory_info().rss)

        return ResourceUsage(
            label=artifact.label,
            baseline_rss_bytes=int(baseline_rss),
            peak_rss_bytes=int(peak_rss),
            cpu_time_ms=max(0.0, cpu_ms),
            wall_time_ms=wall_ms,
            threads=protocol.threads,
            artifact_bytes=artifact.size_bytes,
        )


class BatchScalingBenchmarker:
    """배치 크기를 바꿔 가며 잰다. (실습 4-14)

    같은 결과물, 같은 워밍업, 같은 스레드 수.
    **바뀌는 것은 한 번에 몇 개를 넣느냐뿐이다.**
    """

    def __init__(self, runtimes: RuntimeRegistry) -> None:
        self._runtimes = runtimes

    def scale(
        self,
        artifact: ModelArtifact,
        batch_sizes: tuple[int, ...],
        protocol: MeasurementProtocol,
    ) -> BatchScaling:
        loaded = self._runtimes.require(artifact.artifact_id)

        points: list[BatchPoint] = []
        for size in sorted(set(batch_sizes)):
            sample = np.zeros((size, *loaded.input_shape), dtype="float32")
            for _ in range(protocol.warmup_runs):
                loaded.predict(sample)

            timings = np.empty(protocol.measured_runs, dtype="float64")
            for index in range(protocol.measured_runs):
                started = time.perf_counter()
                loaded.predict(sample)
                timings[index] = (time.perf_counter() - started) * 1000.0

            points.append(
                BatchPoint(
                    batch_size=size,
                    p50_ms=float(np.percentile(timings, 50)),
                    p95_ms=float(np.percentile(timings, 95)),
                )
            )
        return BatchScaling(points=tuple(points))
