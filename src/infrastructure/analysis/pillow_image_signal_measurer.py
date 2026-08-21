"""ImageSignalMeasurer 구현 (Pillow/numpy). — 실습 1-4

이미지에서 세는 것.
    1. 열리지 않는 파일         ← 수집 경로에서 깨진 것
    2. 초점 점수 (Laplacian 분산) ← 흐림
    3. 평균 밝기                 ← 조명 변화
    4. average hash 중복         ← 같은 사진이 여러 번
    5. 해상도 종류 수            ← 카메라/설정이 섞임
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from domain.data.signal import ImageIntegrityMeasurement
from domain.data.source import DataSourceDescriptor
from infrastructure.errors import SourceUnreadable

_SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

_LAPLACIAN_KERNEL = np.array(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype="float64"
)


class PillowImageSignalMeasurer:
    """domain.data.ports.ImageSignalMeasurer 구현."""

    def __init__(self, hash_size: int = 16) -> None:
        """hash_size 는 지문의 해상도다.

        8 로 줄이면 같은 부품을 찍은 서로 다른 사진까지 '중복'으로 뭉뚱그린다.
        주조 부품처럼 생김새가 비슷한 데이터셋에서는 지문을 키워야 한다.
        """
        self._hash_size = hash_size

    def measure(self, source: DataSourceDescriptor) -> ImageIntegrityMeasurement:
        root = Path(source.uri)
        if not root.exists() or not root.is_dir():
            raise SourceUnreadable(
                f"이미지 디렉터리를 찾을 수 없다: {source.uri}", subject=source.uri
            )

        paths = sorted(
            p for p in root.rglob("*") if p.suffix.lower() in _SUPPORTED_SUFFIXES
        )
        if not paths:
            raise SourceUnreadable(
                f"이미지가 한 장도 없다: {source.uri}", subject=source.uri
            )

        unreadable = 0
        focus_scores: list[float] = []
        brightness: list[float] = []
        resolutions: Counter[tuple[int, int]] = Counter()
        hashes: Counter[int] = Counter()

        for path in paths:
            try:
                with Image.open(path) as image:
                    image.load()
                    gray = np.asarray(image.convert("L"), dtype="float64")
                    resolutions[image.size] += 1
                    hashes[self._average_hash(image)] += 1
            except (UnidentifiedImageError, OSError, ValueError):
                unreadable += 1
                continue

            brightness.append(float(gray.mean()))
            focus_scores.append(_focus_score(gray))

        duplicates = sum(count - 1 for count in hashes.values() if count > 1)

        return ImageIntegrityMeasurement(
            total_images=len(paths),
            unreadable_count=unreadable,
            focus_scores=tuple(focus_scores),
            brightness_values=tuple(brightness),
            visual_duplicate_count=duplicates,
            distinct_resolution_count=len(resolutions),
        )

    def _average_hash(self, image: Image.Image) -> int:
        """difference hash — 이웃 화소의 밝기 순서를 지문으로 삼는다.

        평균만 쓰는 average hash 보다 국소 구조를 잘 남겨서,
        '비슷하게 생긴 부품'과 '같은 사진'을 구분할 수 있다.
        """
        size = self._hash_size
        small = np.asarray(
            image.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR),
            dtype="float64",
        )
        bits = (small[:, 1:] > small[:, :-1]).flatten()
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        return value


def _focus_score(gray: np.ndarray) -> float:
    """Laplacian 응답의 분산. 경계가 뚜렷할수록 커진다."""
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    windows = np.lib.stride_tricks.sliding_window_view(gray, (3, 3))
    response = np.tensordot(windows, _LAPLACIAN_KERNEL, axes=((2, 3), (0, 1)))
    return float(response.var())
