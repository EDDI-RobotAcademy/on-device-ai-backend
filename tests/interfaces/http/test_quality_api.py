"""Data Quality API 계약과 오류 변환."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from infrastructure.config.container import DataContainer, DataQualityContainer
from infrastructure.monitoring.event_log import RecordingEventPublisher
from infrastructure.persistence.in_memory_assessment_repository import (
    InMemoryAssessmentRepository,
)
from infrastructure.persistence.in_memory_dataset_repository import (
    InMemoryDatasetRepository,
)
from interfaces.http.app import create_app
from interfaces.http.dependencies.container import (
    get_container,
    get_quality_container,
)
from tests.support import quality_scenario as qs

LABEL_RULES = [
    {
        "label": "FAULT",
        "field_name": "active_power_kw",
        "expected_max": 30.0,
        "description": "보호 계전기가 동작하면 부하가 실제로 차단된다",
    },
    {
        "label": "NORMAL",
        "field_name": "active_power_kw",
        "expected_max": 240.0,
        "description": "정격의 110% 를 넘는 구간을 NORMAL 이라 부를 수 없다",
    },
    {
        "label": "OVERLOAD",
        "field_name": "active_power_kw",
        "expected_min": 192.0,
        "description": "과부하 판정 기준 (SOP-PWR-03)",
    },
]


@pytest.fixture
def client(quality) -> Iterator[TestClient]:
    app = create_app()
    publisher = RecordingEventPublisher()
    data = DataContainer(repository=InMemoryDatasetRepository(), publisher=publisher)
    quality_container = DataQualityContainer(
        datasets=data.repository,
        assessments=InMemoryAssessmentRepository(),
        publisher=publisher,
    )
    app.dependency_overrides[get_container] = lambda: data
    app.dependency_overrides[get_quality_container] = lambda: quality_container

    # 모듈 1 단계는 Use Case 로 미리 통과시켜 둔다. 여기서 볼 것은 품질 API 다.
    qs.prepare_dataset(data, "dirty", quality.dirty)
    qs.prepare_dataset(data, "clean", quality.clean)

    with TestClient(app) as test_client:
        yield test_client


def measure_all(client: TestClient, assessment_id: str) -> None:
    base = f"/quality-assessments/{assessment_id}/dimensions"
    assert client.post(f"{base}/completeness", json={}).status_code == 200
    assert client.post(f"{base}/validity", json={}).status_code == 200
    assert (
        client.post(f"{base}/label-quality", json={"rules": LABEL_RULES}).status_code
        == 200
    )
    assert client.post(f"{base}/balance", json={}).status_code == 200
    assert client.post(f"{base}/noise", json={}).status_code == 200
    assert client.post(f"{base}/uniqueness", json={}).status_code == 200


class TestAssessmentLifecycle:
    def test_평가를_시작하면_201(self, client) -> None:
        response = client.post(
            "/datasets/dirty/quality-assessments", json={"assessment_id": "qa-1"}
        )
        assert response.status_code == 201
        body = response.json()
        assert body["dataset_ref"] == "dirty"
        assert body["status"] == "OPEN"
        assert body["measured_dimensions"] == []

    def test_중복_시작은_409(self, client) -> None:
        client.post(
            "/datasets/dirty/quality-assessments", json={"assessment_id": "qa-1"}
        )
        response = client.post(
            "/datasets/dirty/quality-assessments", json={"assessment_id": "qa-1"}
        )
        assert response.status_code == 409

    def test_없는_Dataset_은_404(self, client) -> None:
        response = client.post(
            "/datasets/없음/quality-assessments", json={"assessment_id": "qa-x"}
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "DATASET_NOT_FOUND"

    def test_없는_평가_조회는_404(self, client) -> None:
        response = client.get("/quality-assessments/없음")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "ASSESSMENT_NOT_FOUND"

    def test_Dataset_별_평가_목록(self, client) -> None:
        client.post(
            "/datasets/dirty/quality-assessments", json={"assessment_id": "qa-1"}
        )
        client.post(
            "/datasets/clean/quality-assessments", json={"assessment_id": "qa-2"}
        )
        body = client.get("/datasets/dirty/quality-assessments").json()
        assert [a["assessment_id"] for a in body] == ["qa-1"]


class TestDimensions:
    def test_결측_검사_응답(self, client) -> None:
        client.post(
            "/datasets/dirty/quality-assessments", json={"assessment_id": "qa-1"}
        )
        body = client.post(
            "/quality-assessments/qa-1/dimensions/completeness", json={}
        ).json()

        assert body["dimension"] == "COMPLETENESS"
        assert body["verdict"] == "FAILED"
        assert 0 <= body["score"] <= 100
        assert {f["code"] for f in body["findings"]} >= {"MISSING_HIDDEN"}

    def test_규칙_없는_라벨_검사는_그_사실을_돌려준다(self, client) -> None:
        client.post(
            "/datasets/dirty/quality-assessments", json={"assessment_id": "qa-1"}
        )
        body = client.post(
            "/quality-assessments/qa-1/dimensions/label-quality", json={}
        ).json()
        assert "LABEL_NO_CONSISTENCY_RULE" in {f["code"] for f in body["findings"]}

    def test_기준을_바꾸면_판정이_바뀐다(self, client) -> None:
        client.post(
            "/datasets/clean/quality-assessments", json={"assessment_id": "qa-2"}
        )
        관대 = client.post(
            "/quality-assessments/qa-2/dimensions/balance",
            json={"max_imbalance_ratio": 100.0},
        ).json()
        엄격 = client.post(
            "/quality-assessments/qa-2/dimensions/balance",
            json={"max_imbalance_ratio": 5.0},
        ).json()
        assert 관대["verdict"] == "PASSED"
        assert 엄격["verdict"] == "PASSED_WITH_WARNINGS"


class TestScoreAndGate:
    def test_점수와_학습_영향을_함께_돌려준다(self, client) -> None:
        client.post(
            "/datasets/dirty/quality-assessments", json={"assessment_id": "qa-1"}
        )
        measure_all(client, "qa-1")
        body = client.post(
            "/quality-assessments/qa-1/score", json={"label_rules": LABEL_RULES}
        ).json()

        assert len(body["dimensions"]) == 6
        assert body["overall_score"] < 80
        assert body["impact"]["total_rows"] == 8640
        assert body["impact"]["usable_rows"] < 8640

    def test_오염된_데이터는_게이트에서_막힌다(self, client) -> None:
        client.post(
            "/datasets/dirty/quality-assessments", json={"assessment_id": "qa-1"}
        )
        measure_all(client, "qa-1")
        body = client.post("/quality-assessments/qa-1/gate", json={}).json()

        assert body["is_ready"] is False
        assert body["verdict"] == "FAILED"
        assert body["blocking_reasons"]

    def test_정리된_데이터는_통과한다(self, client) -> None:
        client.post(
            "/datasets/clean/quality-assessments", json={"assessment_id": "qa-2"}
        )
        measure_all(client, "qa-2")
        body = client.post("/quality-assessments/qa-2/gate", json={}).json()

        assert body["is_ready"] is True
        assert body["grade"] == "A"

    def test_측정_없이_점수를_요청하면_409(self, client) -> None:
        client.post(
            "/datasets/clean/quality-assessments", json={"assessment_id": "qa-2"}
        )
        response = client.post("/quality-assessments/qa-2/score", json={})
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "UNSUPPORTED_OPERATION"

    def test_판정_후_재측정은_409(self, client) -> None:
        client.post(
            "/datasets/clean/quality-assessments", json={"assessment_id": "qa-2"}
        )
        measure_all(client, "qa-2")
        client.post("/quality-assessments/qa-2/gate", json={})

        response = client.post(
            "/quality-assessments/qa-2/dimensions/noise", json={}
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "ILLEGAL_STATE_TRANSITION"

        reopened = client.post(
            "/quality-assessments/qa-2/reopen", json={"reason": "센서 교체 후 재수집"}
        )
        assert reopened.status_code == 200
        assert reopened.json()["status"] == "MEASURING"


class TestRemediationAndComparison:
    def test_근거_없는_조치는_422(self, client) -> None:
        client.post(
            "/datasets/dirty/quality-assessments", json={"assessment_id": "qa-1"}
        )
        measure_all(client, "qa-1")
        response = client.post(
            "/quality-assessments/qa-1/remediations",
            json={
                "kind": "IMPUTE",
                "dimension": "COMPLETENESS",
                "target": "temperature_c",
                "affected_rows": 347,
                "rationale": "",
                "decided_by": "데이터팀",
            },
        )
        assert response.status_code == 422

    def test_조치를_기록하면_재측정_전까지_막힌다(self, client) -> None:
        client.post(
            "/datasets/dirty/quality-assessments", json={"assessment_id": "qa-1"}
        )
        measure_all(client, "qa-1")
        body = client.post(
            "/quality-assessments/qa-1/remediations",
            json={
                "kind": "EXCLUDE_SEGMENT",
                "dimension": "COMPLETENESS",
                "target": "temperature_c / LOT 3개",
                "affected_rows": 259,
                "rationale": "결측이 특정 LOT 에 몰려 있어 보간하면 현실이 조작된다",
                "decided_by": "데이터팀 · 설비운영팀",
            },
        ).json()

        assert body["status"] == "REMEDIATING"
        assert body["unverified_dimensions"] == ["COMPLETENESS"]

        gate = client.post("/quality-assessments/qa-1/gate", json={}).json()
        assert gate["is_ready"] is False
        assert any("재측정하지 않았다" in r for r in gate["blocking_reasons"])

    def test_두_평가를_비교한다(self, client) -> None:
        for dataset_id, assessment_id in (("dirty", "qa-1"), ("clean", "qa-2")):
            client.post(
                f"/datasets/{dataset_id}/quality-assessments",
                json={"assessment_id": assessment_id},
            )
            measure_all(client, assessment_id)
            client.post(
                f"/quality-assessments/{assessment_id}/score",
                json={"label_rules": LABEL_RULES},
            )

        body = client.post(
            "/quality-comparisons",
            json={
                "before_assessment_id": "qa-1",
                "after_assessment_id": "qa-2",
                "before_label": "조치 전",
                "after_label": "조치 후",
            },
        ).json()

        assert body["overall_delta"] > 20
        assert len(body["dimensions"]) == 6
        assert body["after_impact"]["conflicting_rows"] == 0


class TestOpenApi:
    def test_품질_엔드포인트가_문서에_노출된다(self, client) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        expected = {
            "/datasets/{dataset_id}/quality-assessments",
            "/quality-assessments/{assessment_id}",
            "/quality-assessments/{assessment_id}/dimensions/completeness",
            "/quality-assessments/{assessment_id}/dimensions/validity",
            "/quality-assessments/{assessment_id}/dimensions/label-quality",
            "/quality-assessments/{assessment_id}/dimensions/balance",
            "/quality-assessments/{assessment_id}/dimensions/noise",
            "/quality-assessments/{assessment_id}/dimensions/uniqueness",
            "/quality-assessments/{assessment_id}/score",
            "/quality-assessments/{assessment_id}/remediations",
            "/quality-assessments/{assessment_id}/gate",
            "/quality-assessments/{assessment_id}/reopen",
            "/quality-comparisons",
        }
        assert expected <= set(paths)
