"""모듈 2 실습용 전력 데이터 생성기.

모듈 1의 plant_power.py 와 별도로 둔다.
모듈 1의 데이터는 **구조**가 깨진 데이터였고, 여기 데이터는 **내용**이 오염된 데이터다.
둘은 다른 이야기이므로 파일도 다르다.

만들어지는 파일 2개:

    plant_power_quality_clean.csv  기준선. 오염이 없다.
    plant_power_quality_dirty.csv  같은 설비, 같은 구조, 오염 10종.

dirty 파일의 핵심 성질:
    **모듈 1의 구조 검증을 통과한다.**
    스키마도 맞고, 시간축도 깨끗하고, 라벨 정의도 서 있다.
    그런데 학습에 쓰면 망한다. 그것이 모듈 2가 존재하는 이유다.

dirty 에 심어 놓은 오염 (실습에서 이걸 전부 찾아내야 한다):

    Q01  temperature_c 결측 4% — 그중 대부분이 특정 LOT 3개에 몰려 있다 (무작위 결측이 아니다)
    Q02  temperature_c 5% 를 0.0 으로 채워 놓음 — 은폐된 결측. 물리 범위 안이라 안 잡힌다
    Q03  active_power_kw 스파이크 120건 (340~390kW) — 물리 범위 안이지만 통계적으로 극단
    Q04  current_a 급변 30건 — 값 하나만 보면 정상, 직전 표본과 비교해야 보인다
    Q05  FAULT 인데 부하가 정상인 행 30건 — 현장 규칙과 모순되는 라벨
    Q06  FAULT 클래스를 40건까지 축소 — 절대 표본 수 부족
    Q07  voltage_v 에 고주파 잡음 주입 — SNR 붕괴
    Q08  입력 값이 완전히 같은 행 120건 — 타임스탬프는 달라서 모듈 1이 못 잡는다
    Q09  인접 행과 사실상 같은 값 200건 — 센서 홀드/재전송
    Q10  Q08 중 40건은 라벨까지 다름 — 같은 입력에 다른 정답
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_INTERVAL_SECONDS = 10
BATCH_SIZE = 180
CYCLE_SAMPLES = 180
SLOW_CYCLE_SAMPLES = 2880

OVERLOAD_THRESHOLD_KW = 192.0
NORMAL_MAX_KW = 240.0
FAULT_MAX_KW = 30.0

CONDITION_NORMAL = "NORMAL"
CONDITION_OVERLOAD = "OVERLOAD"
CONDITION_FAULT = "FAULT"

PRODUCT_CODES = ("P-14A", "P-22B", "P-30D")


@dataclass(frozen=True, slots=True)
class QualitySample:
    clean: Path
    dirty: Path
    holdout: Path
    """오염되지 않은 **다른 날** 데이터. (실습 3-15)

    "데이터를 고치면 모델이 얼마나 달라지는가"를 재려면
    두 모델을 **같은 시험지**로 채점해야 한다.
    그 시험지는 두 학습 파일 어느 쪽과도 겹치지 않아야 한다.
    """


def write_quality_samples(directory: Path, *, seed: int = 20260813) -> QualitySample:
    directory.mkdir(parents=True, exist_ok=True)

    base = _build_clean(rows=8640, start=datetime(2026, 4, 6, 6, 0, 0), seed=seed)

    clean_path = directory / "plant_power_quality_clean.csv"
    base.to_csv(clean_path, index=False)

    dirty = _contaminate(base, seed=seed)
    dirty_path = directory / "plant_power_quality_dirty.csv"
    dirty.to_csv(dirty_path, index=False)

    # 다른 날, 같은 설비. 오염 없음. 두 모델의 공통 시험지다 (실습 3-15).
    holdout = _build_clean(
        rows=5760, start=datetime(2026, 4, 20, 6, 0, 0), seed=seed + 7
    )
    holdout_path = directory / "plant_power_quality_holdout.csv"
    holdout.to_csv(holdout_path, index=False)

    return QualitySample(clean=clean_path, dirty=dirty_path, holdout=holdout_path)


# ---------------------------------------------------------------------------
# 기준선 — 오염이 없는 데이터
# ---------------------------------------------------------------------------
def _build_clean(*, rows: int, start: datetime, seed: int) -> pd.DataFrame:
    """잡음은 있되 신호에 묻히지 않는 수준의 데이터.

    실제 계측기도 잡음은 있다. 문제는 잡음의 존재가 아니라 **비율**이다.
    """
    rng = np.random.default_rng(seed)
    index = np.arange(rows)

    cycle = np.sin(2 * np.pi * index / CYCLE_SAMPLES)
    slow = np.sin(2 * np.pi * index / SLOW_CYCLE_SAMPLES)

    active_power = 150.0 + 45.0 * cycle + 18.0 * slow + rng.normal(0.0, 1.8, rows)
    power_factor = np.clip(0.92 - 0.0004 * (active_power - 150.0), 0.75, 0.99)
    reactive_power = active_power * np.tan(np.arccos(power_factor))

    # 전압은 부하를 따라 완만하게 흔들린다. 순수 잡음이 아니라 신호가 있다.
    voltage = (
        380.0
        - 0.045 * (active_power - 150.0)
        + 1.5 * slow
        + rng.normal(0.0, 0.12, rows)
    )
    current = active_power * 1000.0 / (np.sqrt(3.0) * voltage * power_factor)
    temperature = (
        26.0 + 0.05 * (active_power - 150.0) + 3.0 * slow + rng.normal(0.0, 0.15, rows)
    )
    spindle = 1500.0 + 60.0 * cycle + rng.normal(0.0, 3.0, rows)

    condition = np.where(
        active_power > OVERLOAD_THRESHOLD_KW, CONDITION_OVERLOAD, CONDITION_NORMAL
    ).astype(object)

    # 진짜 트립은 150건. 트립은 순간이 아니라 '구간'이다 — 10표본(100초) 동안 이어진다.
    # 이 구간성이 중요하다. 한 표본짜리 점프로 만들면 잡음 측정이 그것을 잡음으로 오해한다.
    run_starts = rng.choice(np.arange(50, rows - 20, 60), size=15, replace=False)
    fault_positions = np.sort(
        np.concatenate([np.arange(start, start + 10) for start in run_starts])
    )
    active_power[fault_positions] = rng.uniform(8.0, 22.0, fault_positions.size)
    spindle[fault_positions] = 0.0
    current[fault_positions] = (
        active_power[fault_positions] * 1000.0 / (np.sqrt(3.0) * 380.0 * 0.9)
    )
    reactive_power[fault_positions] = active_power[fault_positions] * 0.45
    temperature[fault_positions] -= 1.5
    condition[fault_positions] = CONDITION_FAULT

    lot = index // BATCH_SIZE
    timestamps = [
        start + timedelta(seconds=int(i) * SAMPLE_INTERVAL_SECONDS) for i in index
    ]

    return pd.DataFrame(
        {
            "timestamp": [t.isoformat(sep=" ") for t in timestamps],
            "meter_id": "PM-MAIN-01",
            "batch_id": [f"Q-{start:%Y%m%d}-{n:03d}" for n in lot],
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
    """2차 작업자의 판단. 모듈 1의 라벨 검사(1-6)가 요구하는 교차 검토 열이다.

    기준이 합의된 뒤이므로 일치율이 높다. 그래도 100%는 아니다 — 사람이 하는 일이다.
    """
    rows = condition.size
    review = np.full(rows, "", dtype=object)
    reviewed = rng.choice(rows, size=1500, replace=False)
    review[reviewed] = condition[reviewed]
    disagree = rng.choice(reviewed, size=60, replace=False)
    review[disagree] = CONDITION_OVERLOAD
    return review


# ---------------------------------------------------------------------------
# 오염
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = (
    "active_power_kw",
    "reactive_power_kvar",
    "current_a",
    "voltage_v",
    "temperature_c",
    "spindle_rpm",
)


def _contaminate(frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 4242)
    frame = frame.copy()
    rows = len(frame)
    lots = frame["batch_id"].to_numpy()

    # Q03 유효전력 스파이크 — 물리 범위(0~400kW) 안이지만 통계적으로 극단
    normal_index = np.flatnonzero(frame["condition"].to_numpy() == CONDITION_NORMAL)
    spike_positions = rng.choice(normal_index, size=120, replace=False)
    frame.loc[spike_positions, "active_power_kw"] = np.round(
        rng.uniform(340.0, 390.0, spike_positions.size), 3
    )

    # Q04 전류 급변 — 값 하나만 보면 정상 범위(0~600A) 안이다
    jump_positions = rng.choice(rows, size=30, replace=False)
    frame.loc[jump_positions, "current_a"] = np.round(
        rng.uniform(470.0, 540.0, jump_positions.size), 3
    )

    # Q05 현장 규칙과 모순되는 라벨 — FAULT 인데 부하가 정상이다
    candidates = np.flatnonzero(frame["condition"].to_numpy() == CONDITION_NORMAL)
    mislabeled = rng.choice(
        np.setdiff1d(candidates, spike_positions), size=30, replace=False
    )
    frame.loc[mislabeled, "condition"] = CONDITION_FAULT

    # Q07 전압 잡음 — 전원 주파수와 샘플링 주파수의 간섭(aliasing).
    # 표본마다 부호가 뒤집히는 톱니가 생긴다. 실제 계측에서 흔한 실패 모드다.
    alternating = 2.2 * np.where(np.arange(rows) % 2 == 0, 1.0, -1.0)
    frame["voltage_v"] = np.round(
        frame["voltage_v"].to_numpy() + alternating + rng.normal(0.0, 0.9, rows), 3
    )

    # Q01 온도 결측 4% — 대부분 특정 LOT 3개에 몰려 있다
    unique_lots = pd.unique(lots)
    hot_lots = rng.choice(unique_lots, size=3, replace=False)
    in_hot_lot = np.isin(lots, hot_lots)
    hot_index = np.flatnonzero(in_hot_lot)
    cold_index = np.flatnonzero(~in_hot_lot)
    missing_total = int(rows * 0.04)
    concentrated = rng.choice(
        hot_index, size=min(int(missing_total * 0.75), hot_index.size), replace=False
    )
    scattered = rng.choice(cold_index, size=missing_total - concentrated.size, replace=False)
    frame.loc[np.concatenate([concentrated, scattered]), "temperature_c"] = np.nan

    # Q02 은폐된 결측 — 비어 있던 값을 0.0 으로 채워 넣었다
    remaining = np.flatnonzero(frame["temperature_c"].notna().to_numpy())
    hidden = rng.choice(remaining, size=int(rows * 0.05), replace=False)
    frame.loc[hidden, "temperature_c"] = 0.0

    # Q09 센서 홀드 — 인접 행이 사실상 같은 값 (5행 × 40구간)
    hold_starts = rng.choice(np.arange(100, rows - 10, 40), size=40, replace=False)
    for start in hold_starts:
        for offset in range(1, 5):
            frame.loc[start + offset, list(FEATURE_COLUMNS)] = frame.loc[
                start, list(FEATURE_COLUMNS)
            ].to_numpy()

    # Q08 입력 값이 완전히 같은 행 — 타임스탬프는 그대로라 모듈 1이 못 잡는다
    source_rows = rng.choice(rows, size=120, replace=False)
    target_rows = rng.choice(
        np.setdiff1d(np.arange(rows), source_rows), size=120, replace=False
    )
    for source, target in zip(source_rows, target_rows, strict=True):
        frame.loc[target, list(FEATURE_COLUMNS)] = frame.loc[
            source, list(FEATURE_COLUMNS)
        ].to_numpy()
        frame.loc[target, "condition"] = frame.loc[source, "condition"]

    # Q10 그중 40건은 라벨까지 다르게 — 같은 입력에 다른 정답
    conflicting = rng.choice(target_rows, size=40, replace=False)
    for position in conflicting:
        current_label = frame.loc[position, "condition"]
        frame.loc[position, "condition"] = (
            CONDITION_OVERLOAD if current_label == CONDITION_NORMAL else CONDITION_NORMAL
        )

    return frame
