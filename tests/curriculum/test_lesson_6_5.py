"""실습 6-5 — SageMaker에서 새로운 모델을 학습시켜라.

    pytest -m lesson_6_5 -s

원격 학습은 로컬 학습(모듈 3)과 세 가지가 다르다.

    입력이 저장소에 있다
    계산이 남의 기계에서 돈다
    **언제 끝나는지 모른다**

세 번째가 결정적이다. `trainer.fit()` 은 돌아올 때까지 기다리면 됐다.
여기서는 기다릴 수 없다 — Command → Job → Status → Result (CLAUDE.md §11).
"""

from __future__ import annotations

import pytest

from domain.fleet.training_job import (
    ComputeSpec,
    RemoteJobStatus,
    RemoteTrainingJob,
    TrainingBudgetPolicy,
)
from domain.shared.errors import InvariantViolation
from tests.support import fleet_scenario as fs
from tests.support import report

pytestmark = pytest.mark.lesson_6_5


def codes(findings) -> set[str]:  # noqa: ANN001
    return {f.code for f in findings}


def job(**overrides) -> RemoteTrainingJob:  # noqa: ANN003
    base: dict[str, object] = dict(
        job_id="j1",
        dataset_uri="s3://lake/datasets/b1/",
        output_uri="s3://artifacts/j1/",
        compute=ComputeSpec(instance_type="ml.m5.large", hourly_cost_usd=0.13),
    )
    base.update(overrides)
    return RemoteTrainingJob(**base)  # type: ignore[arg-type]


def test_제출하고_즉시_돌아온다(fleet_env) -> None:
    report.section("실습 6-5 · SageMaker에서 새로운 모델을 학습시켜라")

    view = fleet_env.submitted
    report.block("제출", view.render())

    assert view.job_id == "train-2026-05-24"
    assert view.dataset_uri.startswith("s3://")
    report.note(
        "**기다리지 않는다.** 제출은 즉시 끝나고, 진행 상황은 따로 물어본다."
    )
    report.note(
        "실제 SageMaker 는 기계를 잡는 데만 몇 분이 걸린다. "
        "(moto 는 즉시 Completed 로 만든다 — 그래서 이 실습에서는 폴링이 한 번에 끝난다)"
    )


def test_상태를_물어본다(fleet_env) -> None:
    view = fleet_env.polled
    report.block("폴링", view.render())

    assert view.is_terminal
    assert view.succeeded
    assert view.artifact_uri.startswith("s3://")
    report.note(
        "끝났으면 **결과물 위치**가 온다. 그것이 6-6 의 입력이 된다."
    )


def test_이_세_줄이_곧_청구서다() -> None:
    cheap = ComputeSpec(
        instance_type="ml.m5.large", max_runtime_seconds=1_800, hourly_cost_usd=0.13
    )
    pricey = ComputeSpec(
        instance_type="ml.p3.8xlarge",
        instance_count=2,
        max_runtime_seconds=14_400,
        hourly_cost_usd=17.6,
    )
    report.block(
        "같은 코드, 다른 기계",
        "\n".join([f"  {cheap.describe()}", f"  {pricey.describe()}"]),
    )
    assert cheap.worst_case_cost_usd < 0.1
    assert pricey.worst_case_cost_usd > 100
    report.note(
        "학습 코드보다 이 세 줄을 먼저 본다. "
        "**클라우드 학습에서 가장 흔한 사고는 정확도가 아니라 청구서다.**"
    )


def test_예산을_넘으면_제출하지_않는다(fleet_env) -> None:
    """제출한 뒤에 멈추면 **이미 과금됐다.**"""
    view = fs.submit_training(
        fleet_env.fleet,
        job_id="train-too-expensive",
        compute=ComputeSpec(
            instance_type="ml.p3.8xlarge",
            instance_count=4,
            max_runtime_seconds=28_800,
            hourly_cost_usd=17.6,
        ),
        policy=TrainingBudgetPolicy(max_cost_usd=20.0),
    )
    report.block("$563 짜리 제출", view.render())

    assert "TRAIN_OVER_BUDGET" in {f.code for f in view.findings}
    assert view.status == RemoteJobStatus.PENDING.value
    report.note("제출 자체를 안 했다. 그래서 청구서가 안 나온다.")


def test_비용을_모르면_경고한다() -> None:
    findings = TrainingBudgetPolicy().inspect_submission(
        job(compute=ComputeSpec(instance_type="ml.m5.large"))
    )
    assert "TRAIN_NO_COST_ESTIMATE" in codes(findings)
    report.note(
        "시간당 비용은 AWS 응답에 없다. **우리 설정에서 온다.** "
        "안 적어 두면 예산 검사를 아예 못 한다."
    )


def test_멈추지_않는_학습은_끝나지_않고_과금만_된다() -> None:
    findings = TrainingBudgetPolicy(max_runtime_seconds=7_200).inspect_submission(
        job(
            compute=ComputeSpec(
                instance_type="ml.m5.large",
                max_runtime_seconds=86_400,
                hourly_cost_usd=0.13,
            )
        )
    )
    assert "TRAIN_RUNTIME_TOO_LONG" in codes(findings)
    report.note("**상한이 곧 안전장치다.** 학습이 수렴 안 해도 24시간 뒤엔 멈춘다.")


def test_실패했는데_이유가_없으면_기록이_아니다() -> None:
    with pytest.raises(InvariantViolation) as caught:
        job(status=RemoteJobStatus.FAILED)
    report.note(str(caught.value))
    report.note(
        "그래서 어댑터가 응답에 이유가 없으면 채워 넣는다 — "
        "'CloudWatch 로그를 확인한다'라도 적어야 다음 사람이 찾아간다."
    )


def test_성공했는데_결과물_위치가_없으면_못_찾는다() -> None:
    with pytest.raises(InvariantViolation):
        job(status=RemoteJobStatus.SUCCEEDED)
    report.note("학습은 끝났는데 어디 있는지 모르는 결과물은 없는 것과 같다.")


def test_끝난_뒤에도_지표를_본다() -> None:
    passed = TrainingBudgetPolicy(min_metrics={"macro_recall": 0.9}).inspect_result(
        job(
            status=RemoteJobStatus.SUCCEEDED,
            artifact_uri="s3://artifacts/j1/model.tar.gz",
            metrics={"macro_recall": 0.94},
        )
    )
    failed = TrainingBudgetPolicy(min_metrics={"macro_recall": 0.9}).inspect_result(
        job(
            status=RemoteJobStatus.SUCCEEDED,
            artifact_uri="s3://artifacts/j1/model.tar.gz",
            metrics={"macro_recall": 0.71},
        )
    )
    assert passed == ()
    assert "TRAIN_METRIC_BELOW_FLOOR" in codes(failed)
    report.note(
        "**학습 성공과 쓸 만한 모델은 다르다.** "
        "SageMaker 는 손실이 안 내려가도 Completed 라고 말한다."
    )


def test_지표가_없으면_좋은지_나쁜지_말할_수_없다() -> None:
    findings = TrainingBudgetPolicy(min_metrics={"macro_recall": 0.9}).inspect_result(
        job(
            status=RemoteJobStatus.SUCCEEDED,
            artifact_uri="s3://artifacts/j1/model.tar.gz",
        )
    )
    assert "TRAIN_METRIC_MISSING" in codes(findings)
    report.note(
        "학습 컨테이너가 지표를 안 뱉으면 이렇게 된다. "
        "**출력 형식을 정해 두는 것이 학습 코드의 일부다.**"
    )


def test_아직_안_끝났으면_판단하지_않는다() -> None:
    findings = TrainingBudgetPolicy().inspect_result(job(status=RemoteJobStatus.RUNNING))
    assert "TRAIN_STILL_RUNNING" in codes(findings)
    report.note("돌고 있는 학습에 대해 '실패'라고 말하지 않는다.")


def test_이유_없이_학습을_멈추지_않는다(fleet_env) -> None:
    from application.shared.errors import ConflictingRequest
    from application.fleet.build_and_train import StopTrainingCommand

    with pytest.raises(ConflictingRequest):
        fleet_env.fleet.stop_training_job().execute(
            StopTrainingCommand(job_id="train-2026-05-24", reason="")
        )
    report.note("이미 쓴 비용이 있다. 무엇을 보고 멈췄는지 남아야 한다.")
