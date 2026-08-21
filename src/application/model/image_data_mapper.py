"""이미지 폴더 → ImageDataRef 번역기 (Anti-Corruption Layer). (실습 3-11)

`training_data_mapper` 의 이미지판이다. 하는 일은 같다.

    "바깥 세계의 사실" → "Model Context 가 아는 말"

다른 점은 게이트다.
표 데이터의 게이트(모듈 1·2)는 결측치·시간축·중복을 본다.
이미지에는 그런 열이 없다. 대신 **장수·폴더·해상도**를 본다.

그래서 이미지에는 이미지의 게이트를 둔다.
재는 것은 Infrastructure(ImageFolderInspector), **판정은 Domain**(ImageReadinessPolicy).
"""

from __future__ import annotations

from domain.model.image_data_ref import (
    ImageDataRef,
    ImageFolderReport,
    ImageReadinessPolicy,
)
from domain.model.ports import ImageFolderInspector
from domain.model.tensor_spec import ImageTensorSpec


def image_data_from(
    inspector: ImageFolderInspector,
    *,
    dataset_ref: str,
    root_uri: str,
    spec: ImageTensorSpec,
    policy: ImageReadinessPolicy,
    split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> tuple[ImageDataRef, ImageFolderReport]:
    """폴더를 재고, 판정하고, 참조를 만든다.

    통과하지 못해도 **번역 자체는 된다.**
    막는 것은 TrainingRun.prepare_images() 의 일이다 — 판단은 Domain 이 한다.
    """
    report = inspector.inspect(root_uri)
    labels = tuple(sorted(name for name, count in report.class_counts.items() if count))

    # 폴더가 하나뿐이면 여기서 InvariantViolation 이 난다.
    # 그건 "소견"이 아니라 애초에 성립하지 않는 문제이기 때문이다.
    ref = ImageDataRef(
        dataset_ref=dataset_ref,
        root_uri=report.root_uri,
        spec=spec,
        class_labels=labels,
        split_ratio=split_ratio,
        readiness_findings=policy.inspect(report),
    )
    return ref, report
