"""실습 1-10 — 학습시키기 전에 데이터부터 검증하라.

    pytest -m lesson_1_10 -s

앞의 아홉 실습을 모아 단 하나의 질문에 답한다.

    "이 데이터로 학습을 시작해도 되는가?"

이 판정은 감이 아니라 재현 가능한 규칙이어야 한다.
그래야 석 달 뒤에 "그때 왜 학습을 시작했나"를 설명할 수 있다.
"""

from __future__ import annotations

import pytest

from application.data.certify_dataset_readiness import (
    CertifyDatasetReadinessCommand,
    ReopenDatasetCommand,
)
from application.data.get_dataset import GetDatasetQuery
from domain.data.inspection import InspectionKind
from domain.shared.errors import IllegalStateTransition
from tests.support import report
from tests.support.scenario import (
    run_full_inspection,
    time_series_readiness_policy,
)

pytestmark = pytest.mark.lesson_1_10


def certify(container, dataset_id, **policy_overrides):  # noqa: ANN001, ANN201
    return container.certify_dataset_readiness().execute(
        CertifyDatasetReadinessCommand(
            dataset_id=dataset_id,
            policy=time_series_readiness_policy(**policy_overrides),
        )
    )


def test_현장에서_받은_원본은_학습_착수를_거부당한다(container, power) -> None:
    report.section("실습 1-10 · 학습시키기 전에 데이터부터 검증하라")

    run_full_inspection(container, "raw", power.raw, power.recent_shifted)
    view = certify(container, "raw")

    report.block("원본 판정", view.render())

    assert view.verdict == "FAILED"
    assert view.is_ready is False
    assert len(view.blocking) >= 5

    blocking_codes = {f.code for f in view.blocking}
    # 아홉 개 실습에서 찾아낸 것들이 여기 전부 모인다.
    assert "BELOW_PHYSICAL_RANGE" in blocking_codes      # 1-2
    assert "SIGNAL_STUCK" in blocking_codes              # 1-4
    assert "TIME_DUPLICATED" in blocking_codes           # 1-5
    assert "LABEL_DISAGREEMENT" in blocking_codes        # 1-6
    assert "REPR_DISTRIBUTION_SHIFTED" in blocking_codes  # 1-9

    state = container.get_dataset().execute(GetDatasetQuery(dataset_id="raw"))
    assert state.status == "REJECTED"

    report.note(
        f"차단 사유 {len(view.blocking)}건. "
        "각 사유에 측정값과 기준이 붙어 있으므로, 무엇을 고쳐야 하는지가 명확하다."
    )


def test_정리된_데이터는_학습_착수_승인을_받는다(container, power) -> None:
    run_full_inspection(container, "curated", power.curated, power.recent_stable)
    view = certify(container, "curated")

    report.block("정리본 판정", view.render())

    assert view.is_ready is True
    assert view.verdict == "PASSED_WITH_WARNINGS"
    assert view.blocking == ()

    state = container.get_dataset().execute(GetDatasetQuery(dataset_id="curated"))
    assert state.status == "READY"

    report.note(
        "경고가 남은 채로 통과했다. "
        "클래스 불균형은 데이터를 고쳐서 없앨 문제가 아니라 알고 대응할 문제다."
    )


def test_검사를_빠뜨리면_통과할_수_없다(container, power) -> None:
    """'검사하지 않았다'는 '문제 없다'가 아니다."""
    from tests.support.scenario import declare_schema, profile, register

    register(container, "half", power.curated)
    profile(container, "half")
    declare_schema(container, "half")

    view = certify(container, "half")
    assert view.verdict == "FAILED"
    assert InspectionKind.PARTITION.value in view.missing_kinds
    assert any(f.code == "READINESS_INSPECTION_MISSING" for f in view.blocking)

    report.block("검사를 건너뛴 경우", view.render())


def test_안전_라인에서는_경고도_막을_수_있다(container, power) -> None:
    """같은 데이터, 다른 기준. 판정 규칙은 현장이 정한다."""
    run_full_inspection(container, "curated", power.curated, power.recent_stable)

    lenient = certify(container, "curated", allow_warnings=True)
    assert lenient.is_ready is True

    container.reopen_dataset().execute(
        ReopenDatasetCommand(dataset_id="curated", reason="안전 등급 재검토")
    )
    strict = certify(container, "curated", allow_warnings=False)
    assert strict.is_ready is False

    report.block(
        "같은 데이터에 대한 두 판정",
        f"  경고 허용   : {lenient.verdict}\n"
        f"  경고 불허   : {strict.verdict}  (차단 {len(strict.blocking)}건)",
    )


def test_판정된_데이터는_몰래_바뀌지_않는다(container, power) -> None:
    run_full_inspection(container, "curated", power.curated, power.recent_stable)
    certify(container, "curated")

    from application.data.inspect_time_axis import InspectTimeAxisCommand
    from domain.data.time_axis import SamplingInterval, TimeAxisPolicy

    with pytest.raises(IllegalStateTransition, match="reopen"):
        container.inspect_time_axis().execute(
            InspectTimeAxisCommand(
                dataset_id="curated",
                policy=TimeAxisPolicy(expected_interval=SamplingInterval(10.0)),
            )
        )

    container.reopen_dataset().execute(
        ReopenDatasetCommand(
            dataset_id="curated", reason="전압 센서 교체 후 재수집분 반영"
        )
    )
    state = container.get_dataset().execute(GetDatasetQuery(dataset_id="curated"))
    assert state.status == "INSPECTED"
    assert state.verdict is None

    report.note(
        "READY 인 데이터를 조용히 수정하는 경로가 없다. "
        "되돌리려면 이유를 남겨야 하고, 그 기록이 감사 근거가 된다."
    )


def test_판정_과정이_이벤트로_남는다(container, power, events) -> None:
    """운영 단계에서 '언제부터 이상해졌나'를 되짚으려면 이 기록이 필요하다."""
    run_full_inspection(container, "curated", power.curated, power.recent_stable)
    certify(container, "curated")

    names = events.names()
    assert "DatasetRegistered" in names
    assert "DataSchemaDeclared" in names
    assert "DatasetPartitioned" in names
    assert "DatasetCertifiedReady" in names
    assert names.count("InspectionRecorded") >= 6

    report.block(
        "발행된 Domain Event",
        "\n".join(f"  {i + 1:>2}. {name}" for i, name in enumerate(names)),
    )
