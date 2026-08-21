"""실습 데이터가 약속한 결함을 실제로 담고 있는가.

실습 데이터가 조용히 달라지면 모든 실습의 정답이 어긋난다.
seed 를 고정했으므로 몇 번을 만들어도 같은 파일이 나와야 한다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from infrastructure.sample_data import write_casting_images, write_plant_power_samples


def test_같은_seed_는_같은_파일을_만든다(tmp_path: Path) -> None:
    first = write_plant_power_samples(tmp_path / "a", seed=123)
    second = write_plant_power_samples(tmp_path / "b", seed=123)
    assert first.raw.read_bytes() == second.raw.read_bytes()


def test_다른_seed_는_다른_파일을_만든다(tmp_path: Path) -> None:
    first = write_plant_power_samples(tmp_path / "a", seed=1)
    second = write_plant_power_samples(tmp_path / "b", seed=2)
    assert first.raw.read_bytes() != second.raw.read_bytes()


def test_원본에_심어_놓은_결함이_전부_들어_있다(power) -> None:
    frame = pd.read_csv(power.raw)

    assert len(frame) == 8460                                   # D07 공백 2구간
    assert frame["temperature_c"].isna().mean() > 0.02          # D01
    assert (frame["current_a"] < 0).sum() == 20                 # D03
    assert frame["meter_id"].nunique() == 1                     # D10
    assert "operator_note" in frame.columns                     # D11
    assert (frame["condition"] == "UNKNOWN").sum() == 6          # D08
    assert frame["condition"].isna().sum() == 12                # D09
    assert frame["timestamp"].duplicated().sum() == 40          # D05


def test_정리본에는_그_결함들이_없다(power) -> None:
    frame = pd.read_csv(power.curated)

    assert frame["temperature_c"].isna().sum() == 0
    assert (frame["current_a"] < 0).sum() == 0
    assert frame["timestamp"].duplicated().sum() == 0
    assert frame["condition"].isna().sum() == 0
    assert set(frame["condition"].unique()) == {"NORMAL", "OVERLOAD", "FAULT"}
    assert "operator_note" not in frame.columns


def test_최근_표본_두_개는_서로_다른_이야기를_한다(power) -> None:
    curated = pd.read_csv(power.curated)
    stable = pd.read_csv(power.recent_stable)
    shifted = pd.read_csv(power.recent_shifted)

    base = curated["temperature_c"].mean()
    assert abs(stable["temperature_c"].mean() - base) < 1.0
    assert shifted["temperature_c"].mean() - base > 7.0

    # 신규 제품 코드는 shifted 에만 있다.
    known = set(curated["product_code"].unique())
    assert set(stable["product_code"].unique()) <= known
    assert set(shifted["product_code"].unique()) - known


def test_이미지_데이터셋의_구성(castings) -> None:
    images = list(castings.root.rglob("*.png"))
    assert len(images) == 51
    assert castings.corrupt_count == 3
    assert castings.duplicate_count == 4
    assert (castings.root / "ok").is_dir()
    assert (castings.root / "ng").is_dir()


def test_이미지_생성도_재현_가능하다(tmp_path: Path) -> None:
    a = write_casting_images(tmp_path / "a", seed=99)
    b = write_casting_images(tmp_path / "b", seed=99)
    first = sorted(p.name for p in a.root.rglob("*.png"))
    second = sorted(p.name for p in b.root.rglob("*.png"))
    assert first == second
    assert (a.root / "ok" / "ok_000.png").read_bytes() == (
        b.root / "ok" / "ok_000.png"
    ).read_bytes()
