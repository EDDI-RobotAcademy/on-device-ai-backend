"""모듈 5 실습용 현장 운영 데이터 생성기.

앞의 세 데이터와 목적이 다르다.

    모듈 1  구조가 깨진 데이터        — 무엇이 잘못됐는지 찾는다
    모듈 2  내용이 오염된 데이터      — 얼마나 오염됐는지 잰다
    모듈 3  두 게이트를 통과한 데이터 — 이걸로 모델을 만든다
    모듈 5  **배포 뒤에 들어오는 것** — 시간이 지나며 변한다

이 데이터에는 오염이 없다. 오염 대신 **시간**이 있다.
4일치가 하루씩 조금씩 달라진다. 그 변화를 잡아내는 것이 모듈 5 전체다.

심어 놓은 사건 (O01~O06):

    O01  3일차부터 온도가 계속 오른다 (여름이 온다)
         → 입력 드리프트. **지속된다** — 이것이 재학습 사유다 (실습 5-7, 5-11)

    O02  2일차 낮 한 창(08~16시)만 온도가 튄다 (도장 부스 도어 개방)
         → 다음 창에서 돌아온다. **재학습 사유가 아니다** (실습 5-4)
         이 둘을 구분하지 못하면 알람이 하루에 열 번 울리고, 사람은 알람을 끈다.

    O03  4일차부터 부하가 전반적으로 올라간다 (냉각 설비 가동)
         → OVERLOAD 예측 급증. 예측 분포 이동 (실습 5-6)

    O04  4일차에 설비 정지가 잦아진다
         → FAULT 예측 급증. 알람 폭주 (실습 5-6, 5-8)

    O05  DEV-02 한 대만 4일차부터 느려진다 (팬 고장 → 발열 → 클럭 저하)
         → **전부가 아니라 한 대다.** 디바이스별로 안 보면 평균에 묻힌다 (실습 5-5)

    O06  정답 라벨은 약 12% 에만 붙는다
         → 현장의 현실이다. 이것 때문에 정확도를 못 재고,
           그래서 분포와 지연시간으로 대신 본다.

지연시간은 이 파일에 없다. **디바이스가 만드는 숫자이기 때문이다.**
`infrastructure/edge/device_simulator.py` 가 만든다.

seed 를 고정하므로 몇 번을 돌려도 같은 파일이 나온다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from infrastructure.sample_data.plant_power_model import (
    CONDITION_FAULT,
    CONDITION_NORMAL,
    CONDITION_OVERLOAD,
    OVERLOAD_THRESHOLD_KW,
    PRODUCT_CODES,
)

SAMPLE_INTERVAL_SECONDS = 10
"""학습 데이터와 **같아야 한다.**

여기를 30초로 바꾸면 어떻게 되는지가 중요하다.
모델은 표본 30개를 한 창으로 본다. 10초 간격이면 그 창은 5분이고,
30초 간격이면 같은 창이 15분이 된다.

    입력 모양은 (30, 6) 으로 똑같다. 오류도 안 난다.
    그런데 모델이 보는 것은 **다른 물리량**이다.

실제로 이렇게 하면 OVERLOAD 예측 비율이 16.7% 에서 1.6% 로 무너진다.
부하 주기가 30분이라 15분짜리 창은 봉우리를 온전히 담지 못하기 때문이다.

현장 게이트웨이가 "트래픽을 줄이려고" 표본을 솎는 순간 이 일이 벌어진다.
**전처리는 모델의 일부다** (실습 5-1) 는 정규화 통계만 뜻하는 것이 아니다.
표본 간격도 모델의 일부다.
"""

CYCLE_SAMPLES = 180
SLOW_CYCLE_SAMPLES = 2_880
BATCH_SIZE = 180

DEVICES = ("DEV-01", "DEV-02", "DEV-03")
DAYS = 4
SAMPLES_PER_DAY = 8_640  # 24시간 / 10초
WINDOW_HOURS = 8
"""관측 창의 길이. **아무 숫자나 쓰면 안 된다.**

이 설비의 느린 부하 주기는 8시간이다.
그보다 짧은 창으로 잘라 보면, 그 창은 주기의 일부만 담는다 —
온도가 오르는 구간만, 또는 내려가는 구간만 들어간다.

그 창의 분포를 학습 전체 분포와 견주면 **드리프트가 없어도 PSI 가 튄다.**
1일차부터 PSI 1.5 가 나오고, 진짜 드리프트가 시작돼도 구분할 수 없게 된다.

    창 4시간 → 1일차 PSI 1.47  (주기의 절반만 봄)
    창 8시간 → 1일차 PSI 0.02  (주기 하나를 온전히 봄)

**관측 창은 공정 주기의 배수여야 한다.**
실습 1-9 에서 최근 표본을 한 주기에 맞췄던 것과 같은 이유다.
"""

WINDOWS_PER_DAY = 24 // WINDOW_HOURS
TOTAL_WINDOWS = DAYS * WINDOWS_PER_DAY

START = datetime(2026, 5, 20, 0, 0, 0)

LABELED_RATIO = 0.12
"""O06 — 현장에서 정답이 붙는 비율."""


@dataclass(frozen=True, slots=True)
class OperationsSample:
    stream: Path
    """4일치 현장 신호. device_id 로 3대가 섞여 있다."""

    window_hours: int = WINDOW_HOURS
    devices: tuple[str, ...] = DEVICES


@dataclass(frozen=True, slots=True)
class DayProfile:
    """하루가 어떻게 달라지는가. **이 표가 심어 놓은 사건의 전부다.**"""

    day: int
    temperature_offset: float
    load_offset: float
    fault_runs: int
    spike_window: int | None = None
    """이 창에서만 온도가 튄다 (0 ~ WINDOWS_PER_DAY-1). O02."""


DAY_PROFILES: tuple[DayProfile, ...] = (
    DayProfile(day=0, temperature_offset=0.0, load_offset=0.0, fault_runs=4),
    DayProfile(day=1, temperature_offset=0.4, load_offset=0.0, fault_runs=4, spike_window=1),
    DayProfile(day=2, temperature_offset=4.5, load_offset=1.0, fault_runs=5),
    DayProfile(day=3, temperature_offset=7.5, load_offset=9.0, fault_runs=14),
)

SPIKE_TEMPERATURE = 5.5
"""O02 — 한 창만 오르고 돌아온다."""


def write_operations_samples(
    directory: Path, *, seed: int = 20260520
) -> OperationsSample:
    directory.mkdir(parents=True, exist_ok=True)

    frames = [
        _build_device(device_id=device, device_index=index, seed=seed + index)
        for index, device in enumerate(DEVICES)
    ]
    stream = pd.concat(frames, ignore_index=True).sort_values(
        ["timestamp", "device_id"], kind="stable"
    )

    path = directory / "plant_power_operations.csv"
    stream.to_csv(path, index=False)
    return OperationsSample(stream=path)


def window_bounds(index: int) -> tuple[str, str]:
    """창 번호(0부터)로 시작·끝 시각을 돌려준다."""
    start = START + timedelta(hours=index * WINDOW_HOURS)
    end = start + timedelta(hours=WINDOW_HOURS) - timedelta(seconds=1)
    return start.isoformat(sep=" "), end.isoformat(sep=" ")


def window_label(index: int) -> str:
    start = START + timedelta(hours=index * WINDOW_HOURS)
    return f"D{index // WINDOWS_PER_DAY + 1}-{start:%H}시"


def _build_device(
    *, device_id: str, device_index: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = [
        _build_day(profile, device_id=device_id, device_index=device_index, rng=rng)
        for profile in DAY_PROFILES
    ]
    return pd.concat(frames, ignore_index=True)


def _build_day(
    profile: DayProfile,
    *,
    device_id: str,
    device_index: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = SAMPLES_PER_DAY
    index = np.arange(rows)
    start = START + timedelta(days=profile.day)

    cycle = np.sin(2 * np.pi * index / CYCLE_SAMPLES)
    slow = np.sin(2 * np.pi * index / SLOW_CYCLE_SAMPLES)

    # 설비마다 부하가 조금씩 다르다. 같은 라인이라도 완전히 같지는 않다.
    line_bias = (device_index - 1) * 1.2

    active_power = (
        150.0
        + profile.load_offset
        + line_bias
        + 50.0 * cycle
        + 6.0 * slow
        + rng.normal(0.0, 1.6, rows)
    )
    power_factor = np.clip(0.92 - 0.0004 * (active_power - 150.0), 0.75, 0.99)
    reactive_power = active_power * np.tan(np.arccos(power_factor))
    voltage = (
        380.0 - 0.045 * (active_power - 150.0) + 1.2 * slow + rng.normal(0.0, 0.12, rows)
    )
    current = active_power * 1000.0 / (np.sqrt(3.0) * voltage * power_factor)

    temperature = (
        26.0
        + profile.temperature_offset
        + 0.05 * (active_power - 150.0)
        + 2.5 * slow
        + rng.normal(0.0, 0.15, rows)
    )
    # O02 — 한 창만 튀고 돌아온다
    if profile.spike_window is not None:
        span = SAMPLES_PER_DAY // WINDOWS_PER_DAY
        begin = profile.spike_window * span
        temperature[begin : begin + span] += SPIKE_TEMPERATURE

    spindle = 1500.0 + 60.0 * cycle + rng.normal(0.0, 3.0, rows)

    condition = np.where(
        active_power > OVERLOAD_THRESHOLD_KW, CONDITION_OVERLOAD, CONDITION_NORMAL
    ).astype(object)

    # O04 — 4일차에 설비 정지가 잦아진다
    spacing = rows // profile.fault_runs
    run_length = 8
    for n in range(profile.fault_runs):
        jitter = int(rng.integers(-spacing // 6, spacing // 6))
        begin = int(np.clip(n * spacing + spacing // 2 + jitter, 40, rows - run_length - 1))
        window = np.arange(begin, begin + run_length)
        active_power[window] = rng.uniform(8.0, 22.0, window.size)
        spindle[window] = 0.0
        current[window] = active_power[window] * 1000.0 / (np.sqrt(3.0) * 380.0 * 0.9)
        reactive_power[window] = active_power[window] * 0.45
        temperature[window] -= 1.5
        condition[window] = CONDITION_FAULT

    lot = index // BATCH_SIZE
    timestamps = [
        start + timedelta(seconds=int(i) * SAMPLE_INTERVAL_SECONDS) for i in index
    ]

    return pd.DataFrame(
        {
            "timestamp": [t.isoformat(sep=" ") for t in timestamps],
            "device_id": device_id,
            "meter_id": f"PM-{device_id[-2:]}",
            "batch_id": [f"O-{start:%Y%m%d}-{n:03d}" for n in lot],
            "product_code": [PRODUCT_CODES[int(n) % len(PRODUCT_CODES)] for n in lot],
            "active_power_kw": np.round(active_power, 3),
            "reactive_power_kvar": np.round(reactive_power, 3),
            "current_a": np.round(current, 3),
            "voltage_v": np.round(voltage, 3),
            "temperature_c": np.round(temperature, 3),
            "spindle_rpm": np.round(spindle, 1),
            "condition": condition,
        }
    )
