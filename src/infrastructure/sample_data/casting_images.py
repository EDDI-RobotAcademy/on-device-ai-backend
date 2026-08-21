"""자동차 다이캐스팅 부품 이미지 생성기.

캡스톤 "자동차 다이캐스팅 부품 이미지 기반 온디바이스 불량 판별"에 대응한다.

디렉터리 구조:
    castings/ok/   양품
    castings/ng/   불량 (표면에 결함 자국이 있다)

심어 놓은 결함 (실습 1-4 에서 찾아내야 한다):
    I01  열리지 않는 파일 3장 (수집 중 깨짐)
    I02  초점이 나간 이미지 6장 (결함이 화면에서 사라진다)
    I03  조명이 바뀐 이미지 16장 (밝기 분포 이동)
    I04  시각적으로 동일한 중복 4장
    I05  해상도가 다른 이미지 5장 (카메라 교체 흔적)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BASE_SIZE = 96
ALT_SIZE = 128


@dataclass(frozen=True, slots=True)
class CastingImageSample:
    root: Path
    ok_count: int
    ng_count: int
    corrupt_count: int
    blurred_count: int
    duplicate_count: int


def write_casting_images(
    directory: Path, *, seed: int = 20260812
) -> CastingImageSample:
    rng = np.random.default_rng(seed)
    root = directory / "castings"
    ok_dir = root / "ok"
    ng_dir = root / "ng"
    for path in (ok_dir, ng_dir):
        path.mkdir(parents=True, exist_ok=True)

    ok_count, ng_count = 26, 18
    written: list[Path] = []

    for i in range(ok_count):
        size = ALT_SIZE if i in (3, 11) else BASE_SIZE
        image = _render_part(rng, size=size, defective=False)
        if 10 <= i < 26:  # I03 조명 변화 — 오후 교대 후 조명을 바꿔 달았다
            image = _shift_exposure(image, +90)
        if i in (7, 19, 23):  # I02 초점 이탈
            image = image.filter(ImageFilter.GaussianBlur(radius=2.4))
        path = ok_dir / f"ok_{i:03d}.png"
        image.save(path)
        written.append(path)

    for i in range(ng_count):
        size = ALT_SIZE if i in (2, 9, 15) else BASE_SIZE
        image = _render_part(rng, size=size, defective=True)
        if i in (4, 12, 16):  # I02 초점 이탈
            image = image.filter(ImageFilter.GaussianBlur(radius=2.4))
        path = ng_dir / f"ng_{i:03d}.png"
        image.save(path)
        written.append(path)

    # I04 중복 — 같은 부품을 두 번 찍어 두 이름으로 저장한 상황
    duplicates = [
        (ok_dir / "ok_000.png", ok_dir / "ok_000_copy.png"),
        (ok_dir / "ok_001.png", ok_dir / "ok_001_copy.png"),
        (ng_dir / "ng_000.png", ng_dir / "ng_000_copy.png"),
        (ng_dir / "ng_001.png", ng_dir / "ng_001_copy.png"),
    ]
    for original, copy in duplicates:
        copy.write_bytes(original.read_bytes())

    # I01 깨진 파일 — 확장자는 png 인데 내용이 이미지가 아니다
    corrupt = [ok_dir / "ok_corrupt_a.png", ng_dir / "ng_corrupt_a.png", ng_dir / "ng_corrupt_b.png"]
    for path in corrupt:
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(rng.integers(0, 255, 96)))

    return CastingImageSample(
        root=root,
        ok_count=ok_count,
        ng_count=ng_count,
        corrupt_count=len(corrupt),
        blurred_count=6,
        duplicate_count=len(duplicates),
    )


def _render_part(rng: np.random.Generator, *, size: int, defective: bool) -> Image.Image:
    """주조 부품 한 장. 금속 질감 + 윤곽 + (불량이면) 표면 결함."""
    texture = rng.normal(128.0, 9.0, (size, size))
    gradient = np.linspace(-14.0, 14.0, size)
    texture += gradient[None, :] + gradient[:, None] * 0.4
    image = Image.fromarray(np.clip(texture, 0, 255).astype("uint8")).convert("RGB")

    draw = ImageDraw.Draw(image)
    margin = size // 8
    draw.ellipse(
        (margin, margin, size - margin, size - margin),
        outline=(196, 196, 196),
        width=max(size // 24, 2),
    )
    draw.rectangle(
        (size // 3, size // 3, size - size // 3, size - size // 3),
        outline=(84, 84, 84),
        width=max(size // 32, 1),
    )

    if defective:
        for _ in range(int(rng.integers(1, 4))):
            x = int(rng.integers(margin, size - margin))
            y = int(rng.integers(margin, size - margin))
            length = int(rng.integers(size // 10, size // 4))
            draw.line(
                (x, y, x + length, y + int(rng.integers(-4, 5))),
                fill=(38, 38, 38),
                width=2,
            )
    return image


def _shift_exposure(image: Image.Image, delta: int) -> Image.Image:
    array = np.asarray(image, dtype="int16") + delta
    return Image.fromarray(np.clip(array, 0, 255).astype("uint8"))
