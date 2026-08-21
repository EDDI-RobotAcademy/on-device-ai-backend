"""공장 실시간 전력 데이터 생성기.

캡스톤 3개 산업군 중 "제조 공장 실시간 전력 데이터 기반 이상 징후 탐지"에 대응한다.

만들어지는 파일 4개:

    plant_power_raw.csv            현장에서 막 받은 파일. 결함이 전부 들어 있다.
    plant_power_curated.csv        같은 설비, 결함을 정리한 뒤의 파일.
    plant_power_recent_shifted.csv 최근 현장 표본. 여름이 되어 분포가 이동했다.
    plant_power_recent_stable.csv  최근 현장 표본. 분포가 유지되고 있다.

raw 에 심어 놓은 결함 (실습에서 이걸 전부 찾아내야 한다):

    D01  temperature_c 결측 3%
    D02  voltage_v 고착 — 600 표본 동안 값이 변하지 않음 (케이블 접촉 불량)
    D03  current_a 음수 20건 — 물리적으로 불가능
    D04  active_power_kw 210.0 에서 잘림 (계측기 포화)
    D05  timestamp 중복 40건
    D06  timestamp 역순 30쌍
    D07  시간 공백 2구간 × 15분
    D08  condition 에 정의되지 않은 값 'UNKNOWN' 6건
    D09  condition 공백 12건
    D10  meter_id 상수 열 (정보량 0)
    D11  operator_note — 스키마에 없는 열
    D12  라벨 교차검토 불일치 12% (작업자 기준이 다름)
    D13  클래스 불균형 NORMAL : OVERLOAD : FAULT ≈ 95 : 4 : 1
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_INTERVAL_SECONDS = 10
BATCH_SIZE = 180  # 30분마다 생산 LOT 이 바뀐다
POWER_SATURATION_KW = 210.0
STUCK_RUN_LENGTH = 600  # 100분 동안 전압이 한 번도 변하지 않는다
SLOW_CYCLE_SAMPLES = 2880  # 8시간 주기의 완만한 부하 변동

PRODUCT_CODES = ("P-14A", "P-22B", "P-30D")
NEW_PRODUCT_CODE = "P-41F"  # 학습 데이터에는 없던 신규 제품

CONDITION_NORMAL = "NORMAL"
CONDITION_OVERLOAD = "OVERLOAD"
CONDITION_FAULT = "FAULT"


@dataclass(frozen=True, slots=True)
class PlantPowerSample:
    raw: Path
    curated: Path
    recent_shifted: Path
    recent_stable: Path


def write_plant_power_samples(directory: Path, *, seed: int = 20260812) -> PlantPowerSample:
    """4개 CSV 를 만들어 경로를 돌려준다."""
    directory.mkdir(parents=True, exist_ok=True)

    raw = _build_base(
        rows=8640,  # 10초 × 8640 = 24시간
        start=datetime(2026, 3, 2, 6, 0, 0),
        seed=seed,
        temperature_offset=0.0,
        power_scale=1.0,
        batch_prefix="B",
    )
    raw = _inject_defects(raw, seed=seed)
    raw_path = directory / "plant_power_raw.csv"
    raw.to_csv(raw_path, index=False)

    curated = _build_base(
        rows=8640,
        start=datetime(2026, 3, 2, 6, 0, 0),
        seed=seed,
        temperature_offset=0.0,
        power_scale=1.0,
        batch_prefix="B",
    )
    curated = _curate(curated, seed=seed)
    curated_path = directory / "plant_power_curated.csv"
    curated.to_csv(curated_path, index=False)

    # 최근 표본은 완만한 부하 주기 1바퀴(8시간)를 정확히 담는다.
    # 주기를 반만 담으면 설비가 멀쩡해도 분포가 달라 보인다 — 그건 드리프트가 아니다.
    shifted = _build_base(
        rows=SLOW_CYCLE_SAMPLES,
        start=datetime(2026, 7, 14, 6, 0, 0),
        seed=seed + 1,
        temperature_offset=9.0,  # 여름
        power_scale=1.18,  # 증산
        batch_prefix="C",
        new_product_ratio=0.4,  # 학습 때 없던 제품 투입
    )
    shifted = _curate(shifted, seed=seed + 1)
    shifted_path = directory / "plant_power_recent_shifted.csv"
    shifted.to_csv(shifted_path, index=False)

    stable = _build_base(
        rows=SLOW_CYCLE_SAMPLES,
        start=datetime(2026, 3, 9, 6, 0, 0),
        seed=seed + 2,
        temperature_offset=0.3,
        power_scale=1.01,
        batch_prefix="B",
    )
    stable = _curate(stable, seed=seed + 2)
    stable_path = directory / "plant_power_recent_stable.csv"
    stable.to_csv(stable_path, index=False)

    return PlantPowerSample(
        raw=raw_path,
        curated=curated_path,
        recent_shifted=shifted_path,
        recent_stable=stable_path,
    )


# ---------------------------------------------------------------------------
# 기본 신호
# ---------------------------------------------------------------------------
def _build_base(
    *,
    rows: int,
    start: datetime,
    seed: int,
    temperature_offset: float,
    power_scale: float,
    batch_prefix: str,
    new_product_ratio: float = 0.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = np.arange(rows)

    timestamps = [
        start + timedelta(seconds=int(i) * SAMPLE_INTERVAL_SECONDS) for i in index
    ]

    # 생산 사이클: 30분 주기 부하 + 8시간 주기 완만한 변동
    cycle = np.sin(2 * np.pi * index / float(BATCH_SIZE))
    slow = np.sin(2 * np.pi * index / float(SLOW_CYCLE_SAMPLES))
    load = 150.0 + 45.0 * cycle + 18.0 * slow + rng.normal(0.0, 6.0, rows)

    # 간헐적 과부하 스파이크
    spike_positions = rng.choice(rows, size=max(rows // 60, 1), replace=False)
    load[spike_positions] += rng.uniform(45.0, 95.0, spike_positions.size)

    active_power = np.clip(load * power_scale, 5.0, None)
    power_factor = np.clip(0.92 - 0.0004 * (active_power - 150.0), 0.75, 0.99)
    reactive_power = active_power * np.tan(np.arccos(power_factor))

    voltage = 380.0 + rng.normal(0.0, 2.2, rows)
    current = active_power * 1000.0 / (np.sqrt(3.0) * voltage * power_factor)

    temperature = (
        26.0
        + temperature_offset
        + 0.012 * (active_power - 150.0)
        + 3.0 * slow
        + rng.normal(0.0, 0.8, rows)
    )
    spindle_rpm = np.where(
        active_power > 60.0, 1500 + 6.0 * cycle * 10 + rng.normal(0, 25, rows), 0.0
    )

    batch_id = [
        f"{batch_prefix}-{start:%Y%m%d}-{i // BATCH_SIZE:03d}" for i in index
    ]

    # 제품 코드는 LOT 단위로 바뀐다. 신규 제품은 뒤쪽 LOT 부터 투입된다.
    lot_numbers = index // BATCH_SIZE
    lot_count = int(lot_numbers.max()) + 1
    new_from_lot = lot_count - int(lot_count * new_product_ratio)
    product_code = np.array(
        [
            NEW_PRODUCT_CODE
            if new_product_ratio > 0 and lot >= new_from_lot
            else PRODUCT_CODES[int(lot) % len(PRODUCT_CODES)]
            for lot in lot_numbers
        ],
        dtype=object,
    )

    condition = np.full(rows, CONDITION_NORMAL, dtype=object)
    condition[active_power > 215.0] = CONDITION_OVERLOAD
    fault_positions = rng.choice(rows, size=max(rows // 100, 1), replace=False)
    condition[fault_positions] = CONDITION_FAULT

    return pd.DataFrame(
        {
            "timestamp": [t.isoformat(sep=" ") for t in timestamps],
            "meter_id": "PM-MAIN-01",
            "batch_id": batch_id,
            "product_code": product_code,
            "active_power_kw": np.round(active_power, 3),
            "reactive_power_kvar": np.round(reactive_power, 3),
            "current_a": np.round(current, 3),
            "voltage_v": np.round(voltage, 3),
            "temperature_c": np.round(temperature, 3),
            "spindle_rpm": np.round(spindle_rpm, 1),
            "condition": condition,
        }
    )


# ---------------------------------------------------------------------------
# 결함 주입 (raw)
# ---------------------------------------------------------------------------
def _inject_defects(frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 977)
    frame = frame.copy()
    rows = len(frame)

    # D04 계측기 포화 — 250kW 를 넘는 값이 전부 250.0 으로 잘린다
    frame["active_power_kw"] = frame["active_power_kw"].clip(upper=POWER_SATURATION_KW)

    # D02 전압 고착 — 케이블 접촉 불량. 마지막 값이 그대로 유지된다.
    stuck_start = 3200
    frame.loc[stuck_start : stuck_start + STUCK_RUN_LENGTH - 1, "voltage_v"] = 379.412

    # D03 전류 음수 — 물리적으로 불가능
    negative_positions = rng.choice(rows, size=20, replace=False)
    frame.loc[negative_positions, "current_a"] = np.round(
        rng.uniform(-3.0, -0.2, 20), 3
    )

    # D01 온도 결측 3%
    missing_positions = rng.choice(rows, size=int(rows * 0.03), replace=False)
    frame.loc[missing_positions, "temperature_c"] = np.nan

    # D08 정의되지 않은 라벨 / D09 라벨 공백
    unknown_positions = rng.choice(rows, size=6, replace=False)
    frame.loc[unknown_positions, "condition"] = "UNKNOWN"
    blank_positions = rng.choice(
        np.setdiff1d(np.arange(rows), unknown_positions), size=12, replace=False
    )
    frame.loc[blank_positions, "condition"] = ""

    # D13 클래스 불균형 강화 — OVERLOAD 대부분을 NORMAL 로 되돌린다
    overload_index = frame.index[frame["condition"] == CONDITION_OVERLOAD]
    if len(overload_index) > 0:
        keep = rng.choice(
            overload_index, size=max(int(len(overload_index) * 0.35), 1), replace=False
        )
        drop = np.setdiff1d(np.asarray(overload_index), keep)
        frame.loc[drop, "condition"] = CONDITION_NORMAL

    # D12 교차 검토 — 400건 중 12% 불일치
    frame["condition_review"] = ""
    review_positions = np.sort(rng.choice(rows, size=400, replace=False))
    frame.loc[review_positions, "condition_review"] = frame.loc[
        review_positions, "condition"
    ]
    disagree = rng.choice(review_positions, size=48, replace=False)
    frame.loc[disagree, "condition_review"] = CONDITION_OVERLOAD

    # D11 스키마에 없는 열
    note = np.full(rows, "", dtype=object)
    note_positions = rng.choice(rows, size=60, replace=False)
    note[note_positions] = rng.choice(
        ["교대 인수인계", "필터 점검", "냉각수 보충", "알람 확인"], size=60
    )
    frame["operator_note"] = note

    # D07 시간 공백 2구간 × 15분(90표본)
    gap_starts = (1500, 6000)
    drop_index: list[int] = []
    for start in gap_starts:
        drop_index.extend(range(start, start + 90))
    frame = frame.drop(index=drop_index).reset_index(drop=True)

    # D05 타임스탬프 중복 40건
    duplicate_positions = rng.choice(
        np.arange(1, len(frame)), size=40, replace=False
    )
    for position in duplicate_positions:
        frame.loc[position, "timestamp"] = frame.loc[position - 1, "timestamp"]

    # D06 역순 30쌍 — 수집 큐가 순서를 보장하지 못한 흔적
    swap_positions = rng.choice(
        np.arange(1, len(frame) - 1, 3), size=30, replace=False
    )
    stamps = frame["timestamp"].to_numpy(copy=True)
    for position in swap_positions:
        stamps[position], stamps[position - 1] = stamps[position - 1], stamps[position]
    frame["timestamp"] = stamps

    return frame


# ---------------------------------------------------------------------------
# 정리본 (curated / recent)
# ---------------------------------------------------------------------------
def _curate(frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """결함 없는 버전.

    데이터를 '고쳤다'는 것은 결측을 채웠다는 뜻이 아니다.
    수집 설정을 바로잡고, 라벨 기준을 합의하고, 못 믿을 구간을 잘라냈다는 뜻이다.
    """
    rng = np.random.default_rng(seed + 613)
    frame = frame.copy()
    rows = len(frame)

    # 라벨 기준 합의 후 재라벨링 — 불균형은 남지만 정의는 명확하다
    condition = frame["condition"].to_numpy(copy=True)
    normal_index = np.flatnonzero(condition == CONDITION_NORMAL)
    promote = rng.choice(normal_index, size=int(rows * 0.06), replace=False)
    condition[promote] = CONDITION_OVERLOAD
    frame["condition"] = condition

    # 교차 검토 — 기준 합의 후 불일치 4%
    review = np.full(rows, "", dtype=object)
    review_positions = np.sort(
        rng.choice(rows, size=min(1500, rows), replace=False)
    )
    review[review_positions] = frame["condition"].to_numpy()[review_positions]
    disagree = rng.choice(review_positions, size=int(len(review_positions) * 0.04), replace=False)
    review[disagree] = CONDITION_FAULT
    frame["condition_review"] = review

    return frame
