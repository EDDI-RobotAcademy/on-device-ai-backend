"""이미지 폴더 → 배열. (실습 3-11)

`torch_materializer.PillowImageTensorMaterializer` 와 같은 세 단계를 쓴다.

    1. 크기 조정   2. 배열 변환   3. 정규화

다른 점은 **분할까지 한다**는 것이다.
그리고 그 분할이 이 파일에서 가장 조심스러운 부분이다.

시계열은 시간 순서로 자른다 (실습 3-8). 이미지에는 시간이 없다.
그래서 **클래스 비율을 유지한 채 섞어서** 나눈다.
그냥 순서대로 자르면 폴더 순서 때문에 test 가 한 클래스로만 채워진다.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from domain.model.image_data_ref import ImageDataRef, ImageFolderReport
from domain.model.tensor_spec import DatasetTensorSummary, ImageTensorSpec, TensorLayout
from infrastructure.errors import SourceUnreadable

SPLITS = ("train", "validation", "test")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True, slots=True)
class LabeledArrays:
    """분할된 배열 묶음.

    `WindowedDataset` 과 같은 이름의 속성을 갖는다.
    그래서 평가 어댑터(TorchModelEvaluator)가 둘을 구분하지 않아도 된다 —
    평가가 알아야 하는 것은 "입력·정답·라벨 순서"뿐이기 때문이다.
    """

    features: Mapping[str, np.ndarray]
    targets: Mapping[str, np.ndarray]
    labels: tuple[str, ...]
    boundary_overlap_samples: int = 0

    @property
    def class_count(self) -> int:
        return len(self.labels)

    def summaries(self) -> dict[str, DatasetTensorSummary]:
        result: dict[str, DatasetTensorSummary] = {}
        for split in SPLITS:
            x, y = self.features[split], self.targets[split]
            result[split] = DatasetTensorSummary(
                split=split,
                sample_count=int(x.shape[0]),
                sample_shape=tuple(int(d) for d in x.shape[1:]),
                class_counts={
                    self.labels[i]: int((y == i).sum()) for i in range(len(self.labels))
                },
                feature_min=float(x.min()) if x.size else None,
                feature_max=float(x.max()) if x.size else None,
                feature_mean=float(x.mean()) if x.size else None,
                feature_std=float(x.std()) if x.size else None,
                nan_count=int(np.isnan(x).sum()) if x.size else 0,
            )
        return result


class PillowImageFolderInspector:
    """domain.model.ports.ImageFolderInspector 구현. (실습 3-11)

    **세기만 한다.** 통과 여부는 여기서 정하지 않는다.
    """

    def inspect(self, root_uri: str) -> ImageFolderReport:
        root = _require_dir(root_uri)

        counts: Counter[str] = Counter()
        unreadable = 0
        sizes: set[tuple[int, int]] = set()
        digests: Counter[str] = Counter()

        for path in _image_paths(root):
            try:
                with Image.open(path) as image:
                    image.load()
                    sizes.add(image.size)
            except (UnidentifiedImageError, OSError, ValueError):
                unreadable += 1
                continue
            counts[path.parent.name] += 1
            digests[hashlib.sha256(path.read_bytes()).hexdigest()] += 1

        if not counts:
            raise SourceUnreadable(
                f"읽을 수 있는 이미지가 없다: {root_uri}", subject=root_uri
            )

        return ImageFolderReport(
            root_uri=str(root),
            class_counts=dict(counts),
            unreadable_count=unreadable,
            distinct_size_count=len(sizes),
            duplicate_count=sum(n - 1 for n in digests.values() if n > 1),
        )


def class_labels_of(root_uri: str) -> tuple[str, ...]:
    """폴더 이름이 곧 라벨이다. 순서를 이름순으로 **고정한다.**

    순서가 바뀌면 혼동 행렬을 비교할 수 없고, 배포된 모델의 출력 해석이 어긋난다.
    """
    root = _require_dir(root_uri)
    labels = sorted(
        p.name for p in root.iterdir() if p.is_dir() and any(_image_paths(p))
    )
    if len(labels) < 2:
        raise SourceUnreadable(
            f"클래스 폴더가 {len(labels)}개다. 분류 문제가 성립하지 않는다.",
            subject=root_uri,
        )
    return tuple(labels)


def build_image_arrays(data: ImageDataRef, *, seed: int = 42) -> LabeledArrays:
    """이미지를 읽어 분할까지 마친다."""
    root = _require_dir(data.root_uri)
    index = {label: i for i, label in enumerate(data.class_labels)}

    arrays: list[np.ndarray] = []
    targets: list[int] = []
    for path in _image_paths(root):
        label = path.parent.name
        if label not in index:
            continue
        array = load_image_array(path, data.spec)
        if array is None:
            continue  # 열리지 않는 장. 게이트가 이미 세어 두었다.
        arrays.append(array)
        targets.append(index[label])

    if not arrays:
        raise SourceUnreadable(
            f"읽을 수 있는 이미지가 없다: {data.root_uri}", subject=data.root_uri
        )

    features = np.stack(arrays).astype("float32")
    labels = np.array(targets, dtype="int64")
    bounds = _stratified_split(labels, data.split_ratio, seed=seed)

    return LabeledArrays(
        features={split: features[idx] for split, idx in bounds.items()},
        targets={split: labels[idx] for split, idx in bounds.items()},
        labels=data.class_labels,
    )


def load_image_array(path: Path, spec: ImageTensorSpec) -> np.ndarray | None:
    """한 장을 명세대로 바꾼다. 열리지 않으면 None.

    이 세 줄이 학습과 배포에서 **똑같이** 돌아야 한다 (실습 3-3).
    그래서 함수 하나로 두고, 학습·평가·디바이스 추론이 모두 이것을 부른다.
    """
    try:
        with Image.open(path) as image:
            image.load()
            converted = image.convert("L" if spec.channels == 1 else "RGB")
            resized = converted.resize(
                (spec.width, spec.height), Image.Resampling.BILINEAR
            )
            array = np.asarray(resized, dtype="float32") / 255.0
    except (UnidentifiedImageError, OSError, ValueError):
        return None

    if spec.channels == 1:
        array = array[:, :, None]
    array = (array - np.array(spec.mean, dtype="float32")) / np.array(
        spec.std, dtype="float32"
    )
    if spec.layout is not TensorLayout.CHANNEL_LAST:
        array = np.transpose(array, (2, 0, 1))
    return np.ascontiguousarray(array, dtype="float32")


# ---------------------------------------------------------------------------
def _stratified_split(
    labels: np.ndarray, ratio: tuple[float, float, float], *, seed: int
) -> dict[str, np.ndarray]:
    """클래스 비율을 유지한 채 나눈다.

    클래스마다 따로 섞고 따로 자른 뒤 합친다.
    이렇게 하지 않으면 적은 클래스가 한 분할에 몰려서
    "test 정확도 100%" 같은 숫자가 나온다 — 그 클래스가 test 에 없기 때문에.
    """
    rng = np.random.default_rng(seed)
    picked: dict[str, list[np.ndarray]] = {split: [] for split in SPLITS}

    for value in np.unique(labels):
        positions = np.flatnonzero(labels == value)
        rng.shuffle(positions)
        total = len(positions)
        train_end = int(round(total * ratio[0]))
        validation_end = train_end + int(round(total * ratio[1]))
        # 어느 분할도 비지 않게 한다. 비면 그 분할의 지표가 거짓이 된다.
        train_end = min(max(train_end, 1), max(total - 2, 1))
        validation_end = min(max(validation_end, train_end + 1), max(total - 1, train_end + 1))
        picked["train"].append(positions[:train_end])
        picked["validation"].append(positions[train_end:validation_end])
        picked["test"].append(positions[validation_end:])

    result: dict[str, np.ndarray] = {}
    for split in SPLITS:
        merged = np.concatenate(picked[split]) if picked[split] else np.array([], dtype=int)
        rng.shuffle(merged)
        result[split] = merged
    return result


def _require_dir(root_uri: str) -> Path:
    root = Path(root_uri)
    if not root.is_dir():
        raise SourceUnreadable(
            f"이미지 디렉터리를 찾을 수 없다: {root_uri}", subject=root_uri
        )
    return root


def _image_paths(root: Path):  # noqa: ANN201
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
