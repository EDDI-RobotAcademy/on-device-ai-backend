"""실습 3-11 — 이미지로 정상과 불량을 가르는 모델을 학습시켜라.

    pytest -m lesson_3_11 -s

실습 3-3 은 이미지를 숫자로 바꾸는 데서 멈췄다. 여기서 그 숫자로 **모델을 만든다.**

이 실습의 핵심은 정확도가 아니라 이것이다.

    같은 데이터, 같은 구조, 같은 학습률.
    마지막 한 줄(공간 축을 접는 방식)만 바꿨는데 정확도가 0.58 → 0.92 로 뛴다.

**작은 결함은 평균에 묻힌다.** 몇 화소짜리 이물은 64×64 를 평균내면 사라진다.
어느 쪽을 쓸지는 코드가 아니라 결함의 생김새가 정한다.
"""

from __future__ import annotations

import pytest

from domain.model.architecture import (
    ArchitectureKind,
    GlobalPooling,
    ModelArchitecture,
)
from domain.model.errors import ShapeMismatch
from domain.model.image_data_ref import (
    ImageFolderReport,
    ImageReadinessPolicy,
)
from domain.shared.errors import IllegalStateTransition, InvariantViolation
from tests.support import image_scenario as isc
from tests.support import report

pytestmark = pytest.mark.lesson_3_11


def test_이미지로_학습한_모델이_불량을_가른다(trained_casting) -> None:
    report.section("실습 3-11 · 이미지로 정상과 불량을 가르는 모델을 학습시켜라")

    evaluation = trained_casting.evaluation
    report.block("다이캐스팅 표면 균열 판별", evaluation.matrix)

    assert evaluation.accuracy > 0.75
    assert evaluation.macro_recall > 0.7
    report.note(
        "정확도가 100% 가 아니다. **불량의 12% 는 일부러 흐리게 만들어 두었다.** "
        "100% 가 나오는 이미지 데이터셋은 대개 문제를 잘못 만든 것이다."
    )


def test_학습_곡선은_시계열과_같은_규칙으로_본다(trained_casting) -> None:
    """재료가 달라도 판단 기준은 같다. LearningPolicy 는 하나뿐이다."""
    curve = trained_casting.curve
    report.block("학습 곡선", curve.table)

    assert curve.status == "COMPLETED"
    assert curve.epoch_count >= 5
    report.note(
        "이미지 학습이라고 다른 Policy 를 만들지 않았다. "
        "'Loss 가 안 떨어진다'와 '외우기 시작했다'는 재료와 무관한 이야기다."
    )


def test_작은_결함은_평균에_묻힌다(trained_food_average, trained_food_max) -> None:
    """이 실습의 본론."""
    average = trained_food_average.evaluation
    maximum = trained_food_max.evaluation

    report.block(
        "같은 데이터, 같은 구조 — 접는 방식만 다르다",
        f"  AVERAGE : accuracy {average.accuracy:.3f}  macro_recall {average.macro_recall:.3f}\n"
        f"  MAX     : accuracy {maximum.accuracy:.3f}  macro_recall {maximum.macro_recall:.3f}",
    )
    report.block("AVERAGE 로 접었을 때", average.matrix)
    report.block("MAX 로 접었을 때", maximum.matrix)

    assert maximum.accuracy > average.accuracy + 0.15
    report.note(
        "이물(foreign)은 몇 화소짜리다. 48×48 을 전부 평균내면 그 몇 화소는 사라진다. "
        "탄자국(burnt)은 넓어서 평균에도 남는다 — **그래서 한쪽만 못 맞힌다.** "
        "혼동 행렬을 봐야 이게 보인다. 정확도 하나로는 안 보인다."
    )


def test_두_구조의_파라미터_수는_똑같다(
    trained_food_average, trained_food_max
) -> None:
    """더 무거워져서 좋아진 것이 아니다."""
    average = trained_food_average.preparation
    maximum = trained_food_max.preparation

    report.block(
        "무게는 그대로다",
        f"  AVERAGE : {average.architecture}\n"
        f"  MAX     : {maximum.architecture}\n"
        f"  입력 모양은 둘 다 {average.input_shape}",
    )
    assert average.input_shape == maximum.input_shape
    report.note(
        "파라미터도 연산량도 같다. **공짜로 얻은 정확도다.** "
        "모듈 4 에서 무게를 줄이기 전에, 줄이지 않고 얻을 수 있는 것부터 본다."
    )


def test_폴더_이름이_곧_라벨이다(industrial_images) -> None:
    from infrastructure.ml.image_dataset import class_labels_of

    labels = class_labels_of(str(industrial_images.food_root))
    report.block("폴더 → 라벨", f"  {labels}")

    assert labels == ("burnt", "foreign", "ok")
    report.note(
        "이름순으로 **고정한다.** 순서가 바뀌면 혼동 행렬을 비교할 수 없고, "
        "배포된 모델의 출력 번호가 다른 것을 가리키게 된다."
    )


def test_폴더가_하나면_분류_문제가_아니다() -> None:
    from domain.model.image_data_ref import ImageDataRef

    with pytest.raises(InvariantViolation, match="클래스가 둘 미만"):
        ImageDataRef(
            dataset_ref="only-ok",
            root_uri="/tmp/only-ok",
            spec=isc.image_spec(),
            class_labels=("ok",),
        )
    report.note(
        "이건 '소견'이 아니라 **불변식 위반**이다. "
        "통과시키고 경고를 다는 것이 아니라, 아예 만들어지지 않게 한다."
    )


def test_장수가_모자라면_게이트가_막는다() -> None:
    policy = ImageReadinessPolicy(min_samples_per_class=30)
    findings = policy.inspect(
        ImageFolderReport(
            root_uri="/tmp/tiny", class_counts={"ok": 120, "ng": 11}
        )
    )
    report.block("이미지 게이트", "\n".join(f"  {f.describe()}" for f in findings))

    codes = {f.code for f in findings}
    assert "IMG_TOO_FEW_SAMPLES" in codes
    assert "IMG_IMBALANCED" in codes
    report.note(
        "표 데이터의 게이트(결측치·시간축)와 항목이 다르다. "
        "이미지에는 **장수와 폴더와 해상도**가 있다."
    )


def test_게이트를_통과하지_못하면_학습을_시작할_수_없다(
    industrial_images,
) -> None:
    container = isc.new_container()
    with pytest.raises(IllegalStateTransition, match="이미지 게이트"):
        isc.prepare(
            container,
            run_id="run-blocked",
            dataset_ref="castings",
            root=industrial_images.casting_root,
            architecture=isc.image_architecture(class_count=2),
            policy=ImageReadinessPolicy(min_samples_per_class=500),
        )
    report.note(
        "모듈 1·2 에서 했던 것과 같은 구조다. "
        "**게이트는 판단이고, 판단은 Domain 이 한다.**"
    )


def test_폴더를_하나_더_만들면_모델도_다시_만들어야_한다(
    industrial_images,
) -> None:
    """클래스 3개짜리 폴더에 2개짜리 모델을 들이대면 준비 단계에서 막힌다."""
    container = isc.new_container()
    with pytest.raises(ShapeMismatch, match="폴더 수"):
        isc.prepare(
            container,
            run_id="run-mismatch",
            dataset_ref="food",
            root=industrial_images.food_root,
            architecture=isc.image_architecture(class_count=2),
        )
    report.note(
        "'불량 유형을 하나 더 나누자'는 현장 결정이 모델 재학습으로 이어지는 지점이다. "
        "여기서 안 막으면 학습은 돌아가고 **배포 후에 한 클래스가 영원히 안 나온다.**"
    )


def test_전처리_계약과_모델_입력이_다르면_막는다(industrial_images) -> None:
    container = isc.new_container()
    wrong = ModelArchitecture(
        kind=ArchitectureKind.CNN2D,
        input_spec=isc.image_spec(size=64).to_tensor_spec(),  # 모델은 64
        class_count=2,
        hidden_channels=(16, 32),
        kernel_size=3,
    )
    with pytest.raises(ShapeMismatch, match="이미지 명세"):
        isc.prepare(
            container,
            run_id="run-shape",
            dataset_ref="castings",
            root=industrial_images.casting_root,
            architecture=wrong,
            size=48,  # 전처리는 48
        )
    report.note(
        "학습 때 48 로 줄이고 배포 때 64 로 줄이면 아무 에러도 안 난다. "
        "**현장에서만 틀린다.** 그래서 계약을 하나로 두고 준비 단계에서 맞춰 본다."
    )


def test_분할은_클래스_비율을_유지한_채_나뉜다(trained_food_max) -> None:
    """이미지에는 시간이 없다. 순서대로 자르면 test 가 한 폴더로만 채워진다."""
    arrays = trained_food_max.container.image_trainer.last_arrays
    summaries = arrays.summaries()

    report.block(
        "분할마다의 클래스 구성",
        "\n".join(
            f"  {split:<11} {summaries[split].sample_count:>4}장  "
            + "  ".join(
                f"{name} {count}"
                for name, count in sorted(summaries[split].class_counts.items())
            )
            for split in ("train", "validation", "test")
        ),
    )

    for split in ("train", "validation", "test"):
        assert summaries[split].class_count == 3, f"{split} 에 빠진 클래스가 있다"
    report.note(
        "세 분할 모두에 세 클래스가 다 있다. 순서대로 잘랐다면 "
        "test 가 'ok' 로만 채워지고 **정확도가 거짓으로 높게** 나왔을 것이다."
    )
