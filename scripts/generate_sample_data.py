"""실습용 현장 데이터를 data/samples/ 에 만든다.

    .venv/bin/python -m scripts.generate_sample_data

테스트는 이 파일을 쓰지 않는다. (테스트는 tmp 디렉터리에 직접 만든다)
사람이 CSV 를 눈으로 열어 보기 위한 경로다. 실습 1-1 은 그것부터 시작한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from infrastructure.sample_data import (  # noqa: E402
    write_casting_images,
    write_industrial_images,
    write_model_samples,
    write_operations_samples,
    write_plant_power_samples,
    write_quality_samples,
)


def main() -> int:
    target = ROOT / "data" / "samples"
    target.mkdir(parents=True, exist_ok=True)

    power = write_plant_power_samples(target)
    quality = write_quality_samples(target)
    model = write_model_samples(target)
    operations = write_operations_samples(target)
    images = write_casting_images(target)
    training_images = write_industrial_images(target)

    print("전력 시계열 — 모듈 1 (구조가 깨진 데이터)")
    for label, path in (
        ("raw            ", power.raw),
        ("curated        ", power.curated),
        ("recent(shifted)", power.recent_shifted),
        ("recent(stable) ", power.recent_stable),
    ):
        size = path.stat().st_size
        print(f"  {label} {path.relative_to(ROOT)}  ({size:,} bytes)")

    print("\n전력 시계열 — 모듈 2 (구조는 멀쩡하고 내용이 오염된 데이터)")
    for label, path in (
        ("clean(기준선)", quality.clean),
        ("dirty(오염)  ", quality.dirty),
        ("holdout(시험지)", quality.holdout),
    ):
        size = path.stat().st_size
        print(f"  {label} {path.relative_to(ROOT)}  ({size:,} bytes)")

    print("\n전력 시계열 — 모듈 3 (두 게이트를 통과한 학습용 데이터)")
    for label, path in (
        ("train(36h)   ", model.train),
        ("field(홀드아웃)", model.field),
    ):
        size = path.stat().st_size
        print(f"  {label} {path.relative_to(ROOT)}  ({size:,} bytes)")

    print("\n전력 시계열 — 모듈 5 (배포 뒤 4일치 현장 신호)")
    size = operations.stream.stat().st_size
    print(
        f"  stream        {operations.stream.relative_to(ROOT)}  ({size:,} bytes)"
    )
    print(
        f"    디바이스 {len(operations.devices)}대 · 관측 창 {operations.window_hours}시간"
    )
    print("    오염은 없다. 대신 하루씩 조금씩 달라진다 (O01~O06).")

    print("\n부품 이미지")
    total = len(list(images.root.rglob('*.png')))
    print(f"  {images.root.relative_to(ROOT)}  (총 {total} 장)")
    print(f"    양품 {images.ok_count} / 불량 {images.ng_count}")
    print(f"    깨진 파일 {images.corrupt_count} / 흐린 이미지 {images.blurred_count} / 중복 {images.duplicate_count}")

    print("\n산업 이미지 — 모듈 3 (학습이 되는 크기)")
    print(
        f"  {training_images.casting_root.relative_to(ROOT)}  "
        f"({training_images.casting_total} 장)  "
        + " / ".join(f"{k} {v}" for k, v in training_images.casting_counts.items())
    )
    print(
        f"  {training_images.food_root.relative_to(ROOT)}  "
        f"({training_images.food_total} 장)  "
        + " / ".join(f"{k} {v}" for k, v in training_images.food_counts.items())
    )
    print("    조명이 장마다 다르다 — **밝기로는 못 가른다.**")
    print("    불량의 12% 는 흐릿하다 — 그래서 정확도가 100% 가 되지 않는다.")

    print("\n먼저 할 일: plant_power_raw.csv 를 그냥 열어 봐라. 스크롤부터 해봐라.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
