"""API 계약과 오류 변환.

확인하는 것 두 가지.
    1. 프론트엔드가 쓸 수 있는 응답 형태가 유지되는가
    2. Domain 예외가 적절한 HTTP 상태로 바뀌는가 (CLAUDE.md §12)

Route 에 Business Logic 이 없다는 것은, 여기 테스트가 '상태 코드와 필드 이름'만
검사해도 충분하다는 사실로 드러난다.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from infrastructure.config.container import DataContainer
from infrastructure.monitoring.event_log import RecordingEventPublisher
from infrastructure.persistence.in_memory_dataset_repository import (
    InMemoryDatasetRepository,
)
from interfaces.http.app import create_app
from interfaces.http.dependencies.container import get_container
from tests.support.scenario import power_schema


@pytest.fixture
def client(power) -> Iterator[TestClient]:
    app = create_app()
    container = DataContainer(
        repository=InMemoryDatasetRepository(),
        publisher=RecordingEventPublisher(),
    )
    app.dependency_overrides[get_container] = lambda: container
    with TestClient(app) as test_client:
        test_client.container = container  # type: ignore[attr-defined]
        yield test_client


def schema_payload() -> dict[str, object]:
    return {
        "fields": [
            {
                "name": f.name,
                "type": f.type.value,
                "role": f.role.value,
                "unit": f.unit,
                "required": f.required,
                "value_range": (
                    None
                    if f.value_range is None
                    else {
                        "minimum": f.value_range.minimum,
                        "maximum": f.value_range.maximum,
                    }
                ),
            }
            for f in power_schema().fields
        ]
    }


def register(client: TestClient, power, dataset_id: str = "raw", curated: bool = False):  # noqa: ANN001, ANN201
    return client.post(
        "/datasets",
        json={
            "dataset_id": dataset_id,
            "name": "3라인 주회로 전력",
            "uri": str(power.curated if curated else power.raw),
            "source_format": "CSV",
            "modality": "TIME_SERIES",
            "collected_from": "LINE-3 / PM-MAIN-01",
        },
    )


class TestHealth:
    def test_상태_확인(self, client: TestClient) -> None:
        assert client.get("/health").json() == {"status": "ok"}


class TestRegistration:
    def test_등록하면_201_과_초기_상태를_돌려준다(self, client, power) -> None:
        response = register(client, power)
        assert response.status_code == 201

        body = response.json()
        assert body["dataset_id"] == "raw"
        assert body["status"] == "REGISTERED"
        assert body["row_count"] is None
        assert body["verdict"] is None

    def test_중복_등록은_409(self, client, power) -> None:
        register(client, power)
        response = register(client, power)
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "CONFLICTING_REQUEST"

    def test_없는_Dataset_조회는_404(self, client) -> None:
        response = client.get("/datasets/없음")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "DATASET_NOT_FOUND"

    def test_형식과_모달리티_조합이_틀리면_422(self, client, power) -> None:
        response = client.post(
            "/datasets",
            json={
                "dataset_id": "bad",
                "name": "x",
                "uri": str(power.raw),
                "source_format": "CSV",
                "modality": "IMAGE",
                "collected_from": "LINE-3",
            },
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "INVARIANT_VIOLATION"


class TestOrdering:
    def test_프로파일_없이_스키마를_선언하면_409(self, client, power) -> None:
        register(client, power)
        response = client.put("/datasets/raw/schema", json=schema_payload())
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "ILLEGAL_STATE_TRANSITION"

    def test_읽을_수_없는_원본은_422(self, client) -> None:
        client.post(
            "/datasets",
            json={
                "dataset_id": "missing",
                "name": "x",
                "uri": "없는파일.csv",
                "source_format": "CSV",
                "modality": "TIME_SERIES",
                "collected_from": "LINE-3",
            },
        )
        response = client.post("/datasets/missing/profile")
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "SOURCE_UNREADABLE"


class TestInspectionFlow:
    def _through_schema(self, client, power, curated: bool = False) -> None:  # noqa: ANN001
        dataset_id = "curated" if curated else "raw"
        register(client, power, dataset_id=dataset_id, curated=curated)
        client.post(f"/datasets/{dataset_id}/profile")
        client.put(f"/datasets/{dataset_id}/schema", json=schema_payload())

    def test_프로파일_응답이_열_단위_사실을_담는다(self, client, power) -> None:
        register(client, power)
        body = client.post("/datasets/raw/profile").json()

        assert body["row_count"] == 8460
        columns = {c["name"]: c for c in body["columns"]}
        assert columns["temperature_c"]["missing_ratio"] > 0.02
        assert columns["meter_id"]["distinct_count"] == 1

    def test_스키마_초안을_받을_수_있다(self, client, power) -> None:
        register(client, power)
        client.post("/datasets/raw/profile")
        body = client.get("/datasets/raw/schema/draft").json()

        assert body["dataset_id"] == "raw"
        assert any(f["name"] == "timestamp" for f in body["fields"])
        assert isinstance(body["undecided_fields"], list)

    def test_스키마_선언_응답은_검사_결과다(self, client, power) -> None:
        self._through_schema(client, power)
        body = client.put("/datasets/raw/schema", json=schema_payload()).json()

        assert body["kind"] == "SCHEMA"
        assert body["verdict"] == "FAILED"
        codes = {f["code"] for f in body["findings"]}
        assert "BELOW_PHYSICAL_RANGE" in codes

    def test_시간축_검사(self, client, power) -> None:
        self._through_schema(client, power)
        body = client.post(
            "/datasets/raw/inspections/time-axis",
            json={"expected_interval_seconds": 10.0},
        ).json()

        assert body["kind"] == "TIME_AXIS"
        assert body["verdict"] == "FAILED"

    def test_시계열_무작위_분할_요청은_422(self, client, power) -> None:
        """계획 자체가 규칙 위반이다."""
        self._through_schema(client, power, curated=True)
        response = client.post(
            "/datasets/curated/partitions",
            json={
                "strategy": "RANDOM",
                "train": 0.7,
                "validation": 0.15,
                "test": 0.15,
                "time_field": "timestamp",
            },
        )
        assert response.status_code == 422
        assert "미래가 학습에 섞인다" in response.json()["error"]["message"]

    def test_전체_흐름을_HTTP_로만_수행하면_READY_에_도달한다(self, client, power) -> None:
        self._through_schema(client, power, curated=True)

        assert client.post(
            "/datasets/curated/inspections/signal", json={}
        ).status_code == 200
        assert client.post(
            "/datasets/curated/inspections/time-axis",
            json={"expected_interval_seconds": 10.0},
        ).status_code == 200

        label_response = client.put(
            "/datasets/curated/label-space",
            json={
                "field_name": "condition",
                "definitions": [
                    {
                        "name": "NORMAL",
                        "meaning": "정격 부하 범위 안에서 운전 중",
                        "decided_by": "설비운영팀 SOP-PWR-03",
                    },
                    {
                        "name": "OVERLOAD",
                        "meaning": "유효전력이 정격의 110% 를 5초 이상 초과",
                        "decided_by": "설비운영팀 SOP-PWR-03",
                    },
                    {
                        "name": "FAULT",
                        "meaning": "보호 계전기가 동작했거나 설비가 정지",
                        "decided_by": "보전팀 고장이력 시스템",
                    },
                ],
            },
        )
        assert label_response.status_code == 200

        partition = client.post(
            "/datasets/curated/partitions",
            json={
                "strategy": "TIME_ORDERED",
                "train": 0.7,
                "validation": 0.15,
                "test": 0.15,
                "time_field": "timestamp",
            },
        ).json()
        assert partition["time_overlap_seconds"] == 0.0

        design = client.put(
            "/datasets/curated/training-spec",
            json={
                "feature_fields": [
                    "active_power_kw",
                    "reactive_power_kvar",
                    "current_a",
                    "voltage_v",
                    "temperature_c",
                    "spindle_rpm",
                ],
                "label_field": "condition",
                "window": {"length": 30, "stride": 30, "interval_seconds": 10.0},
                "normalization": {"method": "ZSCORE"},
                "fit_normalization": True,
            },
        ).json()
        assert design["input_shape"] == [30, 6]
        assert design["normalization_fitted_on"] == "train"

        representativeness = client.post(
            "/datasets/curated/inspections/representativeness",
            json={
                "observed_uri": str(power.recent_stable),
                "observed_collected_from": "LINE-3 / 3월 2주차",
            },
        ).json()
        assert representativeness["inspection"]["verdict"] == "PASSED"

        readiness = client.post(
            "/datasets/curated/readiness",
            json={"required_kinds": None, "allow_warnings": True},
        ).json()
        assert readiness["is_ready"] is True

        state = client.get("/datasets/curated").json()
        assert state["status"] == "READY"
        assert state["input_shape"] == [30, 6]
        assert len(state["inspections"]) >= 6

    def test_READY_인_Dataset_에_검사를_다시_기록하면_409(self, client, power) -> None:
        self.test_전체_흐름을_HTTP_로만_수행하면_READY_에_도달한다(client, power)
        response = client.post(
            "/datasets/curated/inspections/time-axis",
            json={"expected_interval_seconds": 10.0},
        )
        assert response.status_code == 409

        reopened = client.post(
            "/datasets/curated/reopen", json={"reason": "센서 교체 후 재수집"}
        )
        assert reopened.status_code == 200
        assert reopened.json()["status"] == "INSPECTED"

    def test_reopen_은_이유를_요구한다(self, client, power) -> None:
        self._through_schema(client, power, curated=True)
        response = client.post("/datasets/curated/reopen", json={"reason": ""})
        assert response.status_code == 422  # Pydantic 단계에서 걸린다


class TestListing:
    def test_목록_조회(self, client, power) -> None:
        register(client, power, dataset_id="a")
        register(client, power, dataset_id="b")
        body = client.get("/datasets").json()
        assert {d["dataset_id"] for d in body} == {"a", "b"}


class TestOpenApi:
    def test_모든_실습_엔드포인트가_문서에_노출된다(self, client) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        expected = {
            "/datasets",
            "/datasets/{dataset_id}",
            "/datasets/{dataset_id}/profile",
            "/datasets/{dataset_id}/schema",
            "/datasets/{dataset_id}/schema/draft",
            "/datasets/{dataset_id}/inspections/signal",
            "/datasets/{dataset_id}/inspections/time-axis",
            "/datasets/{dataset_id}/inspections/representativeness",
            "/datasets/{dataset_id}/label-space",
            "/datasets/{dataset_id}/training-spec",
            "/datasets/{dataset_id}/partitions",
            "/datasets/{dataset_id}/readiness",
            "/datasets/{dataset_id}/reopen",
        }
        assert expected <= set(paths)
