"""학습에 쓸 수 있는 크기의 산업 이미지 생성기. (실습 3-11, 3-12)

`casting_images.py` 와 목적이 다르다.

    casting_images.py    **결함을 심어 둔** 51장. 실습 1-4 가 그 결함을 찾는다.
    industrial_images.py **학습이 되는** 수백 장. 실습 3-11 이 이것으로 모델을 만든다.

둘을 한 파일에 두지 않는 이유는, 학습용 데이터에 깨진 파일과 중복을 섞어 두면
"모델이 왜 안 나오는가"와 "데이터가 왜 더러운가"가 한 덩어리로 엉키기 때문이다.
품질 문제는 모듈 1·2 에서 이미 다뤘다. 여기서는 **모델**만 본다.

두 캡스톤 주제에 각각 대응한다.

    castings/   자동차 다이캐스팅 부품 — 표면 균열 (ok / ng)
    food/       식품 제조 공정 표면   — 이물·탄자국 (ok / foreign / burnt)

심어 둔 어려움:
    - 조명이 장마다 다르다. **밝기로는 못 가른다.**
    - 불량의 12% 는 흐릿하다. 그래서 정확도가 100% 가 되지 않는다.
      100% 가 나오는 데이터셋은 대개 문제를 잘못 만든 것이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

RENDER_SIZE = 96
"""저장 해상도. 학습 입력(64×64)보다 크게 저장한다 — 현장 카메라가 그렇다."""


@dataclass(frozen=True, slots=True)
class IndustrialImageSample:
    """만들어 둔 이미지 묶음."""

    casting_root: Path
    food_root: Path
    casting_counts: dict[str, int]
    food_counts: dict[str, int]

    @property
    def casting_total(self) -> int:
        return sum(self.casting_counts.values())

    @property
    def food_total(self) -> int:
        return sum(self.food_counts.values())


def write_industrial_images(
    directory: Path, *, seed: int = 20260816
) -> IndustrialImageSample:
    """두 주제의 학습용 이미지를 만든다."""
    casting_counts = _write_castings(directory / "castings-train", seed=seed)
    food_counts = _write_food(directory / "food", seed=seed + 1)
    return IndustrialImageSample(
        casting_root=directory / "castings-train",
        food_root=directory / "food",
        casting_counts=casting_counts,
        food_counts=food_counts,
    )


# ---------------------------------------------------------------------------
# 다이캐스팅 — 캡스톤 주제 1
# ---------------------------------------------------------------------------
def _write_castings(root: Path, *, seed: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    counts = {"ok": 150, "ng": 130}

    for label, total in counts.items():
        folder = root / label
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(total):
            faint = label == "ng" and i % 8 == 3  # 12% 는 흐릿하다
            image = _render_casting(rng, defective=label == "ng", faint=faint)
            image.save(folder / f"{label}_{i:04d}.png")
    return counts


def _render_casting(
    rng: np.random.Generator, *, defective: bool, faint: bool
) -> Image.Image:
    size = RENDER_SIZE

    # 금속 질감 + 조명. 조명은 장마다 다르다 — **밝기는 단서가 아니다.**
    exposure = float(rng.uniform(-28.0, 28.0))
    texture = rng.normal(128.0 + exposure, 10.0, (size, size))
    tilt = np.linspace(-12.0, 12.0, size)
    texture += tilt[None, :] * rng.uniform(-1.0, 1.0) + tilt[:, None] * 0.4
    image = Image.fromarray(np.clip(texture, 0, 255).astype("uint8")).convert("RGB")

    draw = ImageDraw.Draw(image)
    margin = size // 8 + int(rng.integers(-3, 4))
    draw.ellipse(
        (margin, margin, size - margin, size - margin),
        outline=(198, 198, 198),
        width=3,
    )
    draw.rectangle(
        (size // 3, size // 3, size - size // 3, size - size // 3),
        outline=(86, 86, 86),
        width=2,
    )

    if defective:
        # 표면 균열. 위치와 방향은 매번 다르다.
        fill = 96 if faint else 34
        width = 1 if faint else 3
        for _ in range(int(rng.integers(1, 4))):
            x = int(rng.integers(margin, size - margin))
            y = int(rng.integers(margin, size - margin))
            length = int(rng.integers(size // 8, size // 3))
            angle = float(rng.uniform(0, np.pi))
            draw.line(
                (
                    x,
                    y,
                    x + int(length * np.cos(angle)),
                    y + int(length * np.sin(angle)),
                ),
                fill=(fill, fill, fill),
                width=width,
            )

    if rng.random() < 0.15:  # 현장 카메라는 가끔 초점이 흔들린다
        image = image.filter(ImageFilter.GaussianBlur(radius=0.8))
    return image


# ---------------------------------------------------------------------------
# 식품 제조 공정 — 캡스톤 주제 2
# ---------------------------------------------------------------------------
def _write_food(root: Path, *, seed: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    counts = {"ok": 120, "foreign": 100, "burnt": 100}

    for label, total in counts.items():
        folder = root / label
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(total):
            faint = label != "ok" and i % 8 == 5
            image = _render_food(rng, defect=label, faint=faint)
            image.save(folder / f"{label}_{i:04d}.png")
    return counts


def _render_food(
    rng: np.random.Generator, *, defect: str, faint: bool
) -> Image.Image:
    size = RENDER_SIZE
    exposure = float(rng.uniform(-22.0, 22.0))

    # 컨베이어 배경
    background = rng.normal(70.0 + exposure, 6.0, (size, size, 3))
    image = Image.fromarray(np.clip(background, 0, 255).astype("uint8"))

    # 반죽 표면 — 따뜻한 색, 오돌토돌한 질감
    patty = rng.normal(0.0, 7.0, (size, size))
    base = np.stack(
        [
            np.clip(196 + exposure + patty, 0, 255),
            np.clip(158 + exposure + patty, 0, 255),
            np.clip(108 + exposure + patty, 0, 255),
        ],
        axis=-1,
    ).astype("uint8")
    surface = Image.fromarray(base)

    mask = Image.new("L", (size, size), 0)
    radius = size // 2 - int(rng.integers(5, 11))
    center = size // 2 + int(rng.integers(-3, 4))
    ImageDraw.Draw(mask).ellipse(
        (center - radius, center - radius, center + radius, center + radius), fill=255
    )
    image.paste(surface, (0, 0), mask)

    draw = ImageDraw.Draw(image)
    if defect == "foreign":
        # 이물 — 아주 작고 아주 어둡다. 몇 화소짜리다.
        tone = 92 if faint else 26
        for _ in range(int(rng.integers(2, 5))):
            angle = float(rng.uniform(0, 2 * np.pi))
            distance = float(rng.uniform(0, radius * 0.75))
            x = center + int(distance * np.cos(angle))
            y = center + int(distance * np.sin(angle))
            r = 1 if faint else int(rng.integers(2, 4))
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(tone, tone, tone))
    elif defect == "burnt":
        # 탄 자국 — 넓고 경계가 흐리다. 이물과 정반대 성격이다.
        blotch = Image.new("L", (size, size), 0)
        blotch_draw = ImageDraw.Draw(blotch)
        for _ in range(int(rng.integers(1, 3))):
            angle = float(rng.uniform(0, 2 * np.pi))
            distance = float(rng.uniform(0, radius * 0.5))
            x = center + int(distance * np.cos(angle))
            y = center + int(distance * np.sin(angle))
            r = int(rng.integers(6, 12)) if not faint else int(rng.integers(4, 7))
            blotch_draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
        blotch = blotch.filter(ImageFilter.GaussianBlur(radius=3.0))
        # 반죽 밖으로는 번지지 않는다 — 탄 자국은 표면 위에만 있다.
        strength = 0.35 if faint else 0.8
        combined = (
            np.asarray(blotch, dtype="float32")
            * (np.asarray(mask, dtype="float32") / 255.0)
            * strength
        ).astype("uint8")
        darkened = Image.new("RGB", (size, size), (72, 46, 26))
        image.paste(darkened, (0, 0), Image.fromarray(combined, mode="L"))

    if rng.random() < 0.15:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.8))
    return image
