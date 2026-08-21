"""확장 실습 API — 계약과 오류 변환. (실습 1-11, 2-11, 3-11~3-15, 6-12~6-14)

여기서 확인하는 것은 셋이다.
    1. Route 가 얇은가 — 판단이 Domain 에 있는가
    2. Domain 예외가 HTTP 상태로 제대로 바뀌는가 (CLAUDE.md §12)
    3. 소견(findings)이 응답에 함께 나가는가 — 프론트가 그것을 보여줘야 한다
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from infrastructure.config.container import (
    DataContainer,
    DataQualityContainer,
    FleetContainer,
    ModelContainer,
)
from infrastructure.persistence.in_memory_assessment_repository import (
    InMemoryAssessmentRepository,
)
from infrastructure.persistence.in_memory_dataset_repository import (
    InMemoryDatasetRepository,
)
from infrastructure.persistence.in_memory_training_run_repository import (
    InMemoryTrainingRunRepository,
)
from interfaces.http.app import create_app
from interfaces.http.dependencies.container import (
    get_container,
    get_fleet_container,
    get_model_container,
    get_quality_container,
)
from tests.support import quality_scenario as qs


@pytest.fixture
def data_client(model_data) -> Iterator[TestClient]:  # noqa: ANN001
    container = DataContainer(
        repository=InMemoryDatasetRepository(), publisher=None
    )
    quality = DataQualityContainer(datasets=container.repository, publisher=None)
    qs.prepare_dataset(container, "api-sampling", model_data.train)

    app = create_app()
    app.dependency_overrides[get_container] = lambda: container
    app.dependency_overrides[get_quality_container] = lambda: quality
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture
def fleet_client(fleet_bare) -> Iterator[TestClient]:  # noqa: ANN001
    app = create_app()
    app.dependency_overrides[get_fleet_container] = lambda: fleet_bare
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


class Test수집설계:
    def test_비교표와_소견이_함께_온다(self, data_client: TestClient) -> None:
        response = data_client.post(
            "/datasets/api-sampling/sampling-design",
            json={
                "plans": [
                    {"interval_seconds": 10, "retention_days": 30},
                    {"interval_seconds": 60, "retention_days": 30},
                ],
                "target_event_seconds": 60.0,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["plans"]) == 2
        assert body["cheapest_acceptable"].startswith("10초")
        assert any(f["code"] == "SAMPLING_EVENT_LOST" for f in body["findings"])

    def test_설계가_하나면_422_다(self, data_client: TestClient) -> None:
        response = data_client.post(
            "/datasets/api-sampling/sampling-design",
            json={"plans": [{"interval_seconds": 10}]},
        )
        assert response.status_code == 422

    def test_없는_데이터셋은_404_다(self, data_client: TestClient) -> None:
        response = data_client.post(
            "/datasets/없는것/sampling-design",
            json={
                "plans": [
                    {"interval_seconds": 10},
                    {"interval_seconds": 30},
                ]
            },
        )
        assert response.status_code == 404


class Test불균형완화:
    def test_전략마다_잃는_것이_응답에_있다(self, data_client: TestClient) -> None:
        response = data_client.post(
            "/datasets/api-sampling/rebalancing-comparison",
            json={
                "plans": [
                    {"strategy": "NONE"},
                    {"strategy": "OVERSAMPLE"},
                    {"strategy": "OVERSAMPLE", "applied_after_split": False},
                ]
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["outcomes"][1]["duplicated_rows"] > 0
        assert body["outcomes"][2]["verdict"] == "BLOCKED"

    def test_모르는_전략은_422_다(self, data_client: TestClient) -> None:
        response = data_client.post(
            "/datasets/api-sampling/rebalancing-comparison",
            json={"plans": [{"strategy": "MAGIC"}]},
        )
        assert response.status_code == 422


class Test이미지학습:
    def test_준비하면_201_이고_게이트_소견이_함께_온다(
        self, industrial_images
    ) -> None:  # noqa: ANN001
        container = ModelContainer(
            runs=InMemoryTrainingRunRepository(), publisher=None
        )
        app = create_app()
        app.dependency_overrides[get_model_container] = lambda: container
        with TestClient(app) as client:
            response = client.post(
                "/models/image-training-runs",
                json={
                    "run_id": "api-casting",
                    "dataset_ref": "castings",
                    "root_uri": str(industrial_images.casting_root),
                    "spec": {"width": 48, "height": 48, "channels": 3},
                    "class_count": 2,
                    "hidden_channels": [8, 16],
                    "epochs": 2,
                },
            )
        assert response.status_code == 201
        body = response.json()
        assert body["input_shape"] == [3, 48, 48]
        assert "창 없음" in body["windowing"]

    def test_폴더_수가_안_맞으면_409_다(self, industrial_images) -> None:  # noqa: ANN001
        container = ModelContainer(
            runs=InMemoryTrainingRunRepository(), publisher=None
        )
        app = create_app()
        app.dependency_overrides[get_model_container] = lambda: container
        with TestClient(app) as client:
            response = client.post(
                "/models/image-training-runs",
                json={
                    "run_id": "api-mismatch",
                    "dataset_ref": "food",
                    "root_uri": str(industrial_images.food_root),
                    "spec": {"width": 48, "height": 48, "channels": 3},
                    "class_count": 2,
                    "epochs": 1,
                },
            )
        assert response.status_code in (409, 422)


class Test실험기록:
    def test_남기고_다시_읽는다(self, fleet_client: TestClient) -> None:
        created = fleet_client.post(
            "/experiments/api-sweep/trials",
            json={
                "records": [
                    {
                        "trial_id": "win30",
                        "dataset_version": "power-2026-05",
                        "code_version": "abc1234",
                        "parameters": {"window": "30"},
                        "metrics": {"macro_f1": 0.94},
                        "artifact_uri": "",
                    }
                ]
            },
        )
        assert created.status_code == 201

        response = fleet_client.get("/experiments/api-sweep")
        assert response.status_code == 200
        body = response.json()
        assert body["trial_count"] == 1
        assert any(f["code"] == "EXPR_NOT_REPRODUCIBLE" for f in body["findings"])

    def test_시행_식별자가_비면_422_다(self, fleet_client: TestClient) -> None:
        response = fleet_client.post(
            "/experiments/api-sweep/trials",
            json={"records": [{"trial_id": "  "}]},
        )
        assert response.status_code == 422


class Test저장소거버넌스:
    def test_기본값_버킷은_막힌다(self, fleet_client: TestClient) -> None:
        response = fleet_client.get("/storage/governance")
        assert response.status_code == 200
        body = response.json()
        assert body["verdict"] == "BLOCKED"
        assert not body["versioning_enabled"]

    def test_걸고_나면_통과한다(self, fleet_client: TestClient) -> None:
        response = fleet_client.put(
            "/storage/governance",
            json={"versioning": True, "expiration_days": 365},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["versioning_enabled"]
        assert body["encryption_algorithm"] == "AES256"

    def test_공개_허용_한_줄이_막힌다(self, fleet_client: TestClient) -> None:
        response = fleet_client.put(
            "/storage/governance",
            json={
                "statements": [
                    {
                        "sid": "PublicRead",
                        "effect": "Allow",
                        "principal": "*",
                        "actions": ["s3:GetObject"],
                        "resources": ["arn:aws:s3:::ondevice-ai-lake/*"],
                    }
                ]
            },
        )
        assert response.status_code == 200
        assert any(
            f["code"] == "GOV_PUBLIC_STATEMENT" for f in response.json()["findings"]
        )


class Test엔드포인트:
    BODY = {
        "name": "api-endpoint",
        "variants": [
            {
                "name": "AllTraffic",
                "model_reference": "models/line3/v1.tar.gz",
            }
        ],
        "cycle_time_ms": 30.0,
        "network_round_trip_ms": 42.0,
    }

    def test_띄우면_201_이고_막는_소견이_함께_온다(
        self, fleet_client: TestClient
    ) -> None:
        response = fleet_client.post("/endpoints", json=self.BODY)
        assert response.status_code == 201
        body = response.json()
        assert body["status"] == "InService"
        assert body["verdict"] == "BLOCKED"
        codes = {f["code"] for f in body["findings"]}
        assert "EP_CYCLE_TIME_MISSED" in codes
        assert "EP_NO_OFFLINE_FALLBACK" in codes

        assert fleet_client.delete("/endpoints/api-endpoint").status_code == 204

    def test_갈래가_없으면_422_다(self, fleet_client: TestClient) -> None:
        response = fleet_client.post(
            "/endpoints", json={**self.BODY, "variants": []}
        )
        assert response.status_code == 422
