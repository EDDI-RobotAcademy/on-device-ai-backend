"""실습 5-8 — 문제가 발생하면 모델을 격리하라.

    pytest -m lesson_5_8 -s

격리는 롤백이 아니다. **판단을 멈추는 것**이다.

    격리   이 모델의 판단을 더 이상 쓰지 않는다. 설비는 사람이 본다.
    롤백   이전 모델로 되돌린다 (실습 5-10)

격리가 먼저인 이유:
**이전 모델이 더 나으리라는 보장이 없다.**
입력이 변한 것이라면(실습 5-7) 이전 모델도 똑같이 틀린다.
"""

from __future__ import annotations

import pytest

from application.shared.errors import ConflictingRequest
from domain.operations.health import HealthMetric
from domain.operations.incident import IncidentPolicy
from domain.shared.errors import IllegalStateTransition, InvariantViolation
from tests.support import operations_scenario as os5
from tests.support import report

pytestmark = pytest.mark.lesson_5_8


def test_사건은_창마다_근거와_함께_열린다(deployed) -> None:
    report.section("실습 5-8 · 문제가 발생하면 모델을 격리하라")

    from application.operations.respond_to_incident import ListIncidentsQuery

    incidents = deployed.operations.list_incidents().execute(
        ListIncidentsQuery(watch_id=deployed.watch_id)
    )
    report.block(
        "열린 사건",
        "\n".join(f"  {i.summary}" for i in incidents),
    )
    assert incidents
    assert all(i.findings for i in incidents)
    report.note("**근거 없는 사건은 기록하지 않는다.** 소견이 사건의 본문이다.")


def test_격리는_이유_없이_할_수_없다(operations_container) -> None:
    from domain.operations.deployment import Deployment  # noqa: F401
    from domain.operations.identifiers import DeploymentId

    deployment = operations_container.deployments.find_by_id(
        DeploymentId.of(os5.DEPLOYMENT_ID)
    )
    with pytest.raises(InvariantViolation) as caught:
        deployment.quarantine("   ", "2026-05-23 00:00:00")
    report.note(str(caught.value))


def test_이유를_안_주면_최근_관측에서_찾아_온다(operations_container) -> None:
    """사람이 이유를 지어내는 것보다 기계가 근거를 붙이는 편이 정확하다."""
    windows = os5.windows(operations_container)
    os5.observe(operations_container, windows[-1])

    view = os5.quarantine(operations_container)
    report.block("격리", view.render())

    assert view.status == "QUARANTINED"
    assert view.quarantine_reason
    report.note(f"붙은 이유: {view.quarantine_reason[:90]}...")


def test_근거가_없으면_멈추지_않는다(operations_container) -> None:
    windows = os5.windows(operations_container)
    os5.observe(operations_container, windows[0])  # 1일차 — 아무 일도 없다

    with pytest.raises(ConflictingRequest) as caught:
        os5.quarantine(operations_container)
    report.note(str(caught.value))
    report.note(
        "그래도 멈춰야 한다면 이유를 직접 적어야 한다. "
        "현장 판단을 끄는 것은 기록으로 남는 결정이다."
    )


def test_한_창으로_멈출_것인가_지속을_확인할_것인가(deployed, operations_container) -> None:
    """**이 실습에서 실제로 어려운 결정이다.**"""
    windows = os5.windows(operations_container)

    # 2일차 스파이크 창 하나만 보면 — 즉시 격리 권고가 뜬다.
    spike = os5.observe(operations_container, windows[4])
    report.block("스파이크 창 하나만 봤을 때", spike.render()[:600])
    assert spike.quarantine_recommended

    # 그런데 시간선으로 보면 그 창은 이어지지 않았다.
    onset = os5.onset(
        deployed.operations, metric=HealthMetric.INPUT_PSI, threshold=0.2
    )
    assert onset.first_exceeded != onset.sustained_from
    report.note(
        f"한 창만 보면 {onset.first_exceeded} 에 멈춰야 한다. "
        f"3창 연속을 기다리면 {onset.sustained_from} 에 멈춘다."
    )
    report.note(
        "빨리 멈추면 헛알람으로 라인을 세우고, 기다리면 그동안 불량이 나간다. "
        "**둘 다 비용이 있다.** 그 비용을 아는 사람이 기준을 정한다."
    )
    report.note(
        "그래서 IncidentPolicy 는 정책이지 상수가 아니다 — 요청마다 바꿔 넣을 수 있다."
    )


def test_경고가_여럿_겹치면_그것도_사건이다(deployed) -> None:
    """작은 이상이 여럿이면 큰 이상이다."""
    from domain.operations.identifiers import WatchId

    watch = deployed.operations.watches.find_by_id(WatchId.of(deployed.watch_id))
    last = watch.latest

    lenient = IncidentPolicy(quarantine_on_critical=False, quarantine_on_warning_count=4)
    should, reason = lenient.should_quarantine(last)
    report.note(f"CRITICAL 을 무시해도: {'격리' if should else '통과'}")
    if should:
        report.note(reason[:140])
    assert should


def test_격리해도_모델은_그대로_있다(operations_container) -> None:
    windows = os5.windows(operations_container)
    os5.observe(operations_container, windows[-1])
    before = os5.quarantine(operations_container)

    assert before.status == "QUARANTINED"
    assert before.current_version == 1
    report.note(
        "버전은 그대로 v1 이다. **내리지도, 되돌리지도 않았다** — 쓰지 않을 뿐이다."
    )
    report.note(
        "모르는 상태에서 할 수 있는 가장 안전한 일은 멈추는 것이다. "
        "이전 모델이 더 나으리라는 보장이 없기 때문이다."
    )


def test_격리_중에는_새_버전을_밀어_넣을_수_없다(operations_container, deployed) -> None:
    windows = os5.windows(operations_container)
    os5.observe(operations_container, windows[-1])
    os5.quarantine(operations_container)

    with pytest.raises(IllegalStateTransition):
        os5.release(
            operations_container,
            deployed.optimized,
            deployed.trained,
            require_selected=False,
        )
    report.note("원인을 확인해 resume 하거나, rollback 으로 되돌린 뒤에 한다.")


def test_원인을_확인해야_다시_켠다(operations_container) -> None:
    from application.operations.respond_to_incident import ResumeCommand

    windows = os5.windows(operations_container)
    os5.observe(operations_container, windows[-1])
    os5.quarantine(operations_container)

    with pytest.raises(InvariantViolation):
        operations_container.resume_deployment().execute(
            ResumeCommand(deployment_id=os5.DEPLOYMENT_ID, reason="")
        )

    view = operations_container.resume_deployment().execute(
        ResumeCommand(
            deployment_id=os5.DEPLOYMENT_ID,
            reason="DEV-02 팬 교체 완료, 온도 정상 복귀 확인",
        )
    )
    assert view.status == "DEPLOYED"
    report.note("무엇을 확인했는지 없이 현장 판단을 다시 켜지 않는다.")


def test_사건은_해결_방법과_함께_닫는다(deployed, operations_container) -> None:
    from application.operations.respond_to_incident import (
        ListIncidentsQuery,
        ResolveIncidentCommand,
    )

    windows = os5.windows(operations_container)
    os5.observe(operations_container, windows[-1])

    watch_id = f"watch-{os5.DEPLOYMENT_ID}"
    incidents = operations_container.list_incidents().execute(
        ListIncidentsQuery(watch_id=watch_id, only_open=True)
    )
    assert incidents

    with pytest.raises(InvariantViolation):
        operations_container.resolve_incident().execute(
            ResolveIncidentCommand(
                watch_id=watch_id, incident_id=incidents[0].incident_id, resolution=""
            )
        )

    resolved = operations_container.resolve_incident().execute(
        ResolveIncidentCommand(
            watch_id=watch_id,
            incident_id=incidents[0].incident_id,
            resolution="여름 데이터로 재학습 후 v2 배포",
        )
    )
    assert resolved.status == "RESOLVED"
    report.note(
        "무엇을 해서 끝났는지 없으면 같은 일이 또 일어난다. "
        "그 기록이 다음 사람의 출발점이다."
    )
