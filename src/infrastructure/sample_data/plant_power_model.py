"""모듈 3 실습용 학습 데이터 생성기.

모듈 1·2 의 데이터와 별도로 둔다. 목적이 다르기 때문이다.

    모듈 1  구조가 깨진 데이터 — 무엇이 잘못되었는지 찾는 연습
    모듈 2  내용이 오염된 데이터 — 얼마나 오염되었는지 재는 연습
    모듈 3  **두 게이트를 통과한 데이터** — 이걸로 모델을 만드는 연습

만들어지는 파일 2개:

    plant_power_model_train.csv  36시간. 학습·검증·평가로 나눠 쓴다.
    plant_power_model_field.csv  다음 날 12시간. **현장 홀드아웃** (실습 3-10)

현장 홀드아웃이 따로 있는 이유:
    test 분할은 같은 36시간에서 나온다. 같은 날, 같은 조건이다.
    "현장에서 살아남는가"는 **다른 날 데이터**로만 답할 수 있다.

이 데이터에 오염은 없다. 대신 학습이 실제로 어려운 성질을 갖는다.

    M01  FAULT 가 전체의 2% 남짓 — 소수 클래스 (실습 3-9)
    M02  사건이 36시간에 고르게 분포 — 어느 분할에나 들어간다
    M03  OVERLOAD 는 부하 주기의 봉우리에서만 — 시간 패턴을 봐야 맞힌다 (실습 3-4)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_INTERVAL_SECONDS = 10
CYCLE_SAMPLES = 180
SLOW_CYCLE_SAMPLES = 2880
BATCH_SIZE = 180

OVERLOAD_THRESHOLD_KW = 192.0
"""SOP-PWR-03. 모듈 2 의 라벨 규칙과 같은 숫자다 — 같은 설비이기 때문이다."""

FAULT_MAX_KW = 30.0
FAULT_RUN_LENGTH = 12
FAULT_RUN_COUNT = 20

CONDITION_NORMAL = "NORMAL"
CONDITION_OVERLOAD = "OVERLOAD"
CONDITION_FAULT = "FAULT"

PRODUCT_CODES = ("P-14A", "P-22B", "P-30D")


@dataclass(frozen=True, slots=True)
class ModelSample:
    train: Path
    field: Path


def write_model_samples(directory: Path, *, seed: int = 20260814) -> ModelSample:
    directory.mkdir(parents=True, exist_ok=True)

    train = _build(
        rows=12_960,  # 36시간
        start=datetime(2026, 5, 11, 6, 0, 0),
        seed=seed,
        fault_runs=FAULT_RUN_COUNT,
        temperature_offset=0.0,
    )
    train_path = directory / "plant_power_model_train.csv"
    train.to_csv(train_path, index=False)

    field = _build(
        rows=4_320,  # 다음 날 12시간
        start=datetime(2026, 5, 13, 6, 0, 0),
        seed=seed + 1,
        fault_runs=7,
        temperature_offset=1.5,  # 이틀 뒤. 설비는 그대로, 조건은 조금 다르다
    )
    field_path = directory / "plant_power_model_field.csv"
    field.to_csv(field_path, index=False)

    return ModelSample(train=train_path, field=field_path)


def _build(
    *,
    rows: int,
    start: datetime,
    seed: int,
    fault_runs: int,
    temperature_offset: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = np.arange(rows)

    cycle = np.sin(2 * np.pi * index / CYCLE_SAMPLES)
    slow = np.sin(2 * np.pi * index / SLOW_CYCLE_SAMPLES)

    # 봉우리가 항상 과부하 임계를 넘도록 진폭을 잡는다.
    # 그래야 OVERLOAD 가 36시간 내내 고르게 나타나고, 어느 분할에나 들어간다.
    active_power = 150.0 + 50.0 * cycle + 6.0 * slow + rng.normal(0.0, 1.6, rows)
    power_factor = np.clip(0.92 - 0.0004 * (active_power - 150.0), 0.75, 0.99)
    reactive_power = active_power * np.tan(np.arccos(power_factor))
    voltage = (
        380.0 - 0.045 * (active_power - 150.0) + 1.2 * slow + rng.normal(0.0, 0.12, rows)
    )
    current = active_power * 1000.0 / (np.sqrt(3.0) * voltage * power_factor)
    temperature = (
        26.0
        + temperature_offset
        + 0.05 * (active_power - 150.0)
        + 2.5 * slow
        + rng.normal(0.0, 0.15, rows)
    )
    spindle = 1500.0 + 60.0 * cycle + rng.normal(0.0, 3.0, rows)

    condition = np.where(
        active_power > OVERLOAD_THRESHOLD_KW, CONDITION_OVERLOAD, CONDITION_NORMAL
    ).astype(object)

    # M02 — 트립을 36시간에 고르게 배치한다.
    # 무작위로 뿌리면 우연히 한쪽에 몰리고, 그러면 test 분할에 사건이 하나도 없게 된다.
    spacing = rows // fault_runs
    for n in range(fault_runs):
        jitter = int(rng.integers(-spacing // 6, spacing // 6))
        begin = np.clip(n * spacing + spacing // 2 + jitter, 20, rows - FAULT_RUN_LENGTH - 1)
        window = np.arange(begin, begin + FAULT_RUN_LENGTH)
        active_power[window] = rng.uniform(8.0, 22.0, window.size)
        spindle[window] = 0.0
        current[window] = (
            active_power[window] * 1000.0 / (np.sqrt(3.0) * 380.0 * 0.9)
        )
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
            "meter_id": "PM-MAIN-01",
            "batch_id": [f"M-{start:%Y%m%d}-{n:03d}" for n in lot],
            "product_code": [PRODUCT_CODES[int(n) % len(PRODUCT_CODES)] for n in lot],
            "active_power_kw": np.round(active_power, 3),
            "reactive_power_kvar": np.round(reactive_power, 3),
            "current_a": np.round(current, 3),
            "voltage_v": np.round(voltage, 3),
            "temperature_c": np.round(temperature, 3),
            "spindle_rpm": np.round(spindle, 1),
            "condition": condition,
            "condition_review": _cross_review(condition, rng),
        }
    )


def _cross_review(condition: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """2차 작업자의 판단. 모듈 1의 라벨 검사(실습 1-6)가 요구하는 열이다."""
    rows = condition.size
    review = np.full(rows, "", dtype=object)
    reviewed = rng.choice(rows, size=min(2000, rows), replace=False)
    review[reviewed] = condition[reviewed]
    disagree = rng.choice(reviewed, size=int(len(reviewed) * 0.03), replace=False)
    review[disagree] = CONDITION_OVERLOAD
    return review
