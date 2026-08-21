"""실습 6-14 — 실시간 추론 Endpoint를 띄워라.

    pytest -m lesson_6_14 -s

모듈 6 은 지금까지 **모델을 디바이스로 보내는 길**만 봤다.
반대 방향도 있다. 모델을 클라우드에 두고 디바이스가 물어보는 것이다.

    온디바이스   모델이 현장에 있다. 네트워크가 끊겨도 판단한다.
    Endpoint     모델이 클라우드에 있다. 디바이스는 물어본다.

Endpoint 가 유리한 자리는 분명히 있다 — 무거운 모델, 잦은 교체, 실시간이 아닌 작업.
그러나 이 과정의 주제에서는 대개 답이 아니고, 이유가 셋이다.

    1. 네트워크가 끊기면 **라인이 선다**
    2. 왕복 지연은 **모델을 줄여도 안 줄어든다**
    3. **요청이 없어도 시간당 과금된다**

그래서 이 실습은 "띄우는 법"이 아니라 **"띄워도 되는가"**를 판정하는 자리다.

정직하게: moto 의 `invoke_endpoint` 는 고정된 응답을 돌려준다.
실제 추론이 도는 것이 아니다.
"""

from __future__ import annotations

import pytest

from application.fleet.govern_storage import (
    DeployEndpointCommand,
    TeardownEndpointCommand,
)
from domain.fleet.endpoint import (
    EndpointSpec,
    EndpointVariant,
    OnlineInferenceProfile,
)
from domain.shared.errors import InvariantViolation
from tests.support import report

pytestmark = pytest.mark.lesson_6_14

VARIANT = EndpointVariant(
    name="AllTraffic",
    model_reference="models/line3/v1.3.0/model.tar.gz",
    instance_type="ml.m5.large",
    instance_count=1,
    hourly_cost_usd=0.115,
)
SPEC = EndpointSpec(
    name="line3-inspection", variants=(VARIANT,), purpose="라벨 재검증"
)


def _profile(**overrides):  # noqa: ANN003, ANN202
    base = dict(
        cycle_time_ms=30.0,
        network_round_trip_ms=42.0,
        inference_ms=6.0,
        offline_tolerance_minutes=0.0,
        requests_per_hour=30,
    )
    base.update(overrides)
    return OnlineInferenceProfile(**base)


def test_엔드포인트를_실제로_띄운다(fleet_bare) -> None:
    report.section("실습 6-14 · 실시간 추론 Endpoint를 띄워라")

    view = fleet_bare.deploy_endpoint().execute(
        DeployEndpointCommand(spec=SPEC, profile=_profile())
    )
    report.block("엔드포인트", view.render())

    assert view.status == "InService"
    assert view.variants[0][0] == "AllTraffic"
    report.note(
        "create_model → create_endpoint_config → create_endpoint. "
        "**세 번의 진짜 boto3 요청이다** — API 이름이 틀리면 여기서 터진다."
    )
    fleet_bare.teardown_endpoint().execute(
        TeardownEndpointCommand(name=SPEC.name)
    )


def test_왕복_지연은_모델을_줄여도_안_줄어든다(fleet_bare) -> None:
    """이 실습의 본론 (1)."""
    view = fleet_bare.deploy_endpoint().execute(
        DeployEndpointCommand(spec=SPEC, profile=_profile())
    )

    report.block(
        "사이클 타임",
        f"  왕복 42ms + 추론 6ms = {view.total_latency_ms:g}ms\n"
        f"  사이클 타임 {view.cycle_time_ms:g}ms",
    )

    assert "EP_CYCLE_TIME_MISSED" in [f.code for f in view.findings]
    assert view.verdict == "BLOCKED"
    report.note(
        "모듈 4 에서 지연시간을 0.003ms 까지 줄였다. "
        "**그 노력이 왕복 42ms 앞에서 의미가 없어진다.** "
        "이 문제는 경량화로 못 푼다 — 구조로 푸는 문제다."
    )
    fleet_bare.teardown_endpoint().execute(TeardownEndpointCommand(name=SPEC.name))


def test_네트워크가_끊기면_라인이_선다(fleet_bare) -> None:
    """이 실습의 본론 (2)."""
    view = fleet_bare.deploy_endpoint().execute(
        DeployEndpointCommand(
            spec=SPEC, profile=_profile(cycle_time_ms=5000.0)
        )
    )

    codes = [f.code for f in view.findings]
    report.block("소견", "\n".join(f"  {f.describe()}" for f in view.findings))

    assert "EP_NO_OFFLINE_FALLBACK" in codes
    report.note(
        "사이클 타임이 5초라 지연은 문제가 안 된다. "
        "**그래도 막힌다** — 정전, 스위치 교체, 공사. 공장 회선은 끊긴다. "
        "온디바이스 모델을 함께 두지 않으면 그 시간 동안 "
        "라인이 서거나 검사 없이 흘러간다. **이 과정이 온디바이스인 이유다.**"
    )
    fleet_bare.teardown_endpoint().execute(TeardownEndpointCommand(name=SPEC.name))


def test_요청이_없어도_시간당_과금된다(fleet_bare) -> None:
    """이 실습의 본론 (3)."""
    big = EndpointSpec(
        name="line3-heavy",
        variants=(
            EndpointVariant(
                name="AllTraffic",
                model_reference="models/heavy.tar.gz",
                instance_type="ml.g4dn.xlarge",
                instance_count=2,
                hourly_cost_usd=0.906,
            ),
        ),
    )
    view = fleet_bare.deploy_endpoint().execute(
        DeployEndpointCommand(
            spec=big, profile=_profile(cycle_time_ms=5000.0, requests_per_hour=30)
        )
    )

    report.block(
        "비용",
        f"  인스턴스 {view.instance_count}대\n"
        f"  월 ${view.monthly_cost_usd:,.0f}\n"
        f"  시간당 요청 30건 → 인스턴스당 15건",
    )

    codes = [f.code for f in view.findings]
    assert "EP_ALWAYS_ON_COST" in codes
    assert "EP_UNDERUSED" in codes
    report.note(
        "**분당 한 건짜리 일에 상시 GPU 서버 두 대를 두고 있다.** "
        "요청이 하나도 없는 밤에도 그대로 나간다. "
        "띄워 두고 잊은 엔드포인트가 그렇게 생긴다 — "
        "그래서 실습 끝에 반드시 teardown 을 부른다."
    )
    fleet_bare.teardown_endpoint().execute(TeardownEndpointCommand(name=big.name))


def test_인스턴스가_한_대면_그_한_대가_전부다(fleet_bare) -> None:
    view = fleet_bare.deploy_endpoint().execute(
        DeployEndpointCommand(
            spec=SPEC,
            profile=_profile(
                cycle_time_ms=5000.0,
                offline_tolerance_minutes=30.0,
                requests_per_hour=10_000,
            ),
        )
    )

    assert "EP_SINGLE_INSTANCE" in [f.code for f in view.findings]
    report.note(
        "**그 한 대가 재시작되면 전부 멈춘다.** "
        "현장이 이걸 의존하고 있다면 최소 두 대여야 한다 — "
        "그리고 그 순간 비용이 두 배가 된다."
    )
    fleet_bare.teardown_endpoint().execute(TeardownEndpointCommand(name=SPEC.name))


def test_실습이_끝나면_반드시_지운다(fleet_bare) -> None:
    fleet_bare.deploy_endpoint().execute(
        DeployEndpointCommand(spec=SPEC, profile=_profile())
    )
    fleet_bare.teardown_endpoint().execute(TeardownEndpointCommand(name=SPEC.name))

    from botocore.exceptions import ClientError

    with pytest.raises(ClientError):
        fleet_bare.endpoints.describe(SPEC.name)
    report.note(
        "**지우지 않으면 계속 과금된다.** "
        "교육 계정에서 가장 자주 나는 사고가 이것이다 — "
        "실습이 끝난 엔드포인트가 한 달을 돈다."
    )


def test_클라우드에서는_가중치로_AB_를_한다(fleet_bare) -> None:
    """실습 5-14 의 A/B 가 클라우드에서는 이렇게 생겼다."""
    ab = EndpointSpec(
        name="line3-ab",
        variants=(
            EndpointVariant(
                name="blue",
                model_reference="models/line3/v1.2.0/model.tar.gz",
                instance_type="ml.m5.large",
                weight=0.9,
            ),
            EndpointVariant(
                name="green",
                model_reference="models/line3/v1.3.0/model.tar.gz",
                instance_type="ml.m5.large",
                weight=0.1,
            ),
        ),
    )
    view = fleet_bare.deploy_endpoint().execute(
        DeployEndpointCommand(
            spec=ab,
            profile=_profile(cycle_time_ms=5000.0, offline_tolerance_minutes=30.0),
        )
    )
    report.block("A/B 배분", "\n".join(f"  {n} {w:g}" for n, w in view.variants))

    shifted = fleet_bare.endpoints.shift_traffic(
        ab.name, {"blue": 0.5, "green": 0.5}
    )
    report.block("가중치를 옮긴 뒤", "\n".join(f"  {n} {w:g}" for n, w in shifted.variants))

    assert dict(shifted.variants)["green"] == pytest.approx(0.5)
    report.note(
        "디바이스에서는 OTA 로 웨이브를 나눠야 했다 (실습 6-8). "
        "**클라우드에서는 숫자 하나를 바꾸면 된다.** "
        "그 편함이 엔드포인트의 진짜 장점이고, "
        "실습 5-14 의 규율(멈춤 기준·최소 표본)은 여기서도 그대로 필요하다."
    )
    fleet_bare.teardown_endpoint().execute(TeardownEndpointCommand(name=ab.name))


def test_설정_교체는_moto_에_없다(fleet_bare) -> None:
    """확인하지 못한 경로를 숨기지 않고 적어 둔다."""
    import boto3

    fleet_bare.deploy_endpoint().execute(
        DeployEndpointCommand(spec=SPEC, profile=_profile())
    )
    with pytest.raises(NotImplementedError):
        boto3.client("sagemaker", region_name="ap-northeast-2").update_endpoint(
            EndpointName=SPEC.name, EndpointConfigName=f"{SPEC.name}-config"
        )

    report.note(
        "가중치 변경(`update_endpoint_weights_and_capacities`)은 moto 가 지원한다 — 위 테스트가 그걸 쓴다. "
        "**설정 자체를 갈아끼우는 `update_endpoint` 는 없다.** "
        "새 모델 버전을 넣는 무중단 교체는 그래서 실계정에서 검증해야 한다. "
        "확인하지 못한 것을 확인한 척하지 않는다."
    )
    fleet_bare.teardown_endpoint().execute(TeardownEndpointCommand(name=SPEC.name))


def test_갈래가_없으면_엔드포인트가_아니다() -> None:
    with pytest.raises(InvariantViolation, match="갈래가 하나도 없다"):
        EndpointSpec(name="empty", variants=())
    with pytest.raises(InvariantViolation, match="중복"):
        EndpointSpec(name="dup", variants=(VARIANT, VARIANT))
