"""실습 6-12 — 실험을 기록하지 않으면 다시 만들 수 없다.

    pytest -m lesson_6_12 -s

실습 3-14 는 실험을 **비교**했다. 그건 한 사람의 노트북 안에서였다.
클라우드에서 학습을 돌리기 시작하면 문제가 달라진다.

    학습 잡이 30개 돈다. 아티팩트는 S3 에 있다.
    파라미터는 각자의 노트북에 있다. 두 달 뒤 그 사람이 퇴사한다.

남는 것은 `s3://.../model.tar.gz` 파일 하나다.
**그 파일이 무엇으로 만들어졌는지 아무도 모른다.**

재현에 필요한 넷 — 데이터 · 코드 · 설정 · 결과 — 중 하나라도 없으면
그 실험은 다시 만들 수 없고, 다시 만들 수 없는 모델은 고칠 수도 없다.
"""

from __future__ import annotations

import pytest

from application.fleet.govern_storage import (
    RecordExperimentCommand,
    ReviewExperimentCommand,
)
from domain.fleet.experiment_record import ExperimentRecord
from domain.fleet.object_key import ObjectKey
from domain.shared.errors import InvariantViolation
from tests.support import report

pytestmark = pytest.mark.lesson_6_12

EXPERIMENT = "window-length-sweep"


def _key(path: str) -> ObjectKey:
    """평평한 경로를 ObjectKey 로 감싼다 — 저장소 Port 는 키 객체를 받는다."""
    prefix, _, filename = path.rpartition("/")
    return ObjectKey(prefix=prefix, partitions=(), filename=filename)


def _record(trial: str, **overrides):  # noqa: ANN003, ANN202
    base = dict(
        experiment_id=EXPERIMENT,
        trial_id=trial,
        dataset_version="power-2026-05-11",
        code_version="a1b2c3d",
        parameters={"window": "30", "stride": "30", "lr": "3e-3", "seed": "42"},
        metrics={"macro_f1": 0.94, "accuracy": 0.96, "latency_p95_ms": 0.003},
        artifact_uri="",
        created_at="2026-05-23T09:00:00",
    )
    base.update(overrides)
    return ExperimentRecord(**base)


@pytest.fixture
def store(fleet_bare):  # noqa: ANN001, ANN201
    return fleet_bare


def test_학습_한_번의_전모를_S3_에_남긴다(store) -> None:
    report.section("실습 6-12 · 실험을 기록하지 않으면 다시 만들 수 없다")

    artifact = "experiments/window-length-sweep/artifacts/win30.tar.gz"
    store.store.put(_key(artifact), b"model-bytes")

    keys = store.record_experiment().execute(
        RecordExperimentCommand(
            records=(
                _record("win30", artifact_uri=f"s3://ondevice-ai-lake/{artifact}"),
            )
        )
    )
    report.block("남은 자리", "\n".join(f"  {k}" for k in keys))

    assert keys[0].startswith(f"experiments/{EXPERIMENT}/trials/")
    report.note(
        "노트북이 아니라 **S3 에** 둔다. 사람이 바뀌어도 남아야 하기 때문이다. "
        "키 구조는 실습 6-2 의 규칙을 그대로 따른다 — "
        "실험을 앞에 두면 '이 실험의 시행 전부'가 접두어로 좁혀진다."
    )


def test_기록을_읽어_비교표를_만든다(store) -> None:
    for trial, window, f1 in (("win15", "15", 0.91), ("win30", "30", 0.94)):
        key = f"experiments/{EXPERIMENT}/artifacts/{trial}.tar.gz"
        store.store.put(_key(key), b"model")
        store.record_experiment().execute(
            RecordExperimentCommand(
                records=(
                    _record(
                        trial,
                        parameters={
                            "window": window,
                            "stride": window,
                            "lr": "3e-3",
                            "seed": "42",
                        },
                        metrics={"macro_f1": f1, "accuracy": f1 + 0.02},
                        artifact_uri=f"s3://ondevice-ai-lake/{key}",
                    ),
                )
            )
        )

    view = store.review_experiment().execute(
        ReviewExperimentCommand(experiment_id=EXPERIMENT)
    )
    report.block("실험 대장", view.render())

    assert view.trial_count == 2
    assert view.best_trial_id == "win30"
    assert view.reproducible_count == 2
    report.note(
        "**결과만이 아니라 조건이 함께 있다.** "
        "두 달 뒤 '그때 그 0.94' 를 물으면 이 표가 답한다."
    )


def test_넷_중_하나라도_없으면_다시_만들_수_없다(store) -> None:
    """이 실습의 본론."""
    key = f"experiments/{EXPERIMENT}/artifacts/mystery.tar.gz"
    store.store.put(_key(key), b"model")
    store.record_experiment().execute(
        RecordExperimentCommand(
            records=(
                _record(
                    "mystery",
                    code_version="",  # 어떤 커밋이었는지 없다
                    artifact_uri=f"s3://ondevice-ai-lake/{key}",
                ),
            )
        )
    )

    view = store.review_experiment().execute(
        ReviewExperimentCommand(experiment_id=EXPERIMENT)
    )
    report.block("소견", "\n".join(f"  {f.describe()}" for f in view.findings))

    assert "EXPR_NOT_REPRODUCIBLE" in [f.code for f in view.findings]
    report.note(
        "지표는 다 있다. **코드 버전 한 줄이 없다.** "
        "그러면 같은 데이터, 같은 파라미터로 돌려도 다른 숫자가 나올 수 있고 — "
        "왜 다른지 영영 알 수 없다."
    )


def test_기록만_있고_파일이_없는_경우가_실제로_생긴다(store) -> None:
    store.record_experiment().execute(
        RecordExperimentCommand(
            records=(
                _record(
                    "ghost",
                    artifact_uri="s3://ondevice-ai-lake/experiments/gone.tar.gz",
                ),
            )
        )
    )

    view = store.review_experiment().execute(
        ReviewExperimentCommand(experiment_id=EXPERIMENT)
    )

    assert "ghost" in view.missing_artifacts
    assert "EXPR_ARTIFACT_MISSING" in [f.code for f in view.findings]
    report.note(
        "**S3 에 실제로 있는지 확인한다.** 기록만 믿으면 안 된다 — "
        "수명 주기 규칙이 지웠거나(실습 6-13), 다른 계정으로 옮겼을 수 있다. "
        "그러면 그 기록은 종이다."
    )


def test_데이터가_다른데_안_적으면_모델을_비교한_것이_아니다(store) -> None:
    for trial, version in (("mayA", "power-2026-05"), ("junB", "power-2026-06")):
        key = f"experiments/{EXPERIMENT}/artifacts/{trial}.tar.gz"
        store.store.put(_key(key), b"model")
        store.record_experiment().execute(
            RecordExperimentCommand(
                records=(
                    _record(
                        trial,
                        dataset_version=version,
                        artifact_uri=f"s3://ondevice-ai-lake/{key}",
                    ),
                )
            )
        )

    view = store.review_experiment().execute(
        ReviewExperimentCommand(experiment_id=EXPERIMENT)
    )

    assert "EXPR_DATA_VERSION_MIXED" in [f.code for f in view.findings]
    report.note(
        "6월 데이터가 더 깨끗해서 좋아진 것을 '모델을 개선했다'고 보고하는 일은 "
        "**현장에서 실제로 일어난다** (실습 3-14). "
        "클라우드에서는 데이터도 잡마다 다르기 쉬워서 더 자주 일어난다."
    )


def test_실험과_시행_식별자가_없으면_기록이_아니다() -> None:
    with pytest.raises(InvariantViolation, match="trial_id"):
        ExperimentRecord(experiment_id="e", trial_id="  ")


def test_진짜_S3_요청이다(store) -> None:
    key = f"experiments/{EXPERIMENT}/artifacts/real.tar.gz"
    store.store.put(_key(key), b"model")
    keys = store.record_experiment().execute(
        RecordExperimentCommand(
            records=(_record("real", artifact_uri=f"s3://ondevice-ai-lake/{key}"),)
        )
    )

    body = store.store.get(_key(keys[0]))
    report.block("실제로 올라간 내용", f"  {body[:120].decode('utf-8')}…")
    assert b"dataset_version" in body
    report.note(
        "moto 안에서 돌지만 **요청은 진짜 boto3 가 만든다.** "
        "가짜 클라이언트를 세워 두고 '호출됐다'만 확인하면 API 이름이 틀려도 통과한다."
    )
