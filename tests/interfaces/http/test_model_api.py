"""Model API — Job 흐름과 계약. (CLAUDE.md §11)

핵심은 **학습이 HTTP 요청을 붙잡지 않는다**는 것이다.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from infrastructure.config.container import (
    DataContainer,
    DataQualityContainer,
    ModelContainer,
)
from infrastructure.monitoring.event_log import RecordingEventPublisher
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
    get_model_container,
    get_quality_container,
)
from tests.support import model_scenario as ms

PREPARE_BODY = {
    "run_id": "run-api",
    "dataset_id": "api-ds",
    "assessment_id": "api-qa",
    "architecture": {
        "kind": "CNN1D",
        "input_shape": [30, 6],
        "class_count": 3,
        "hidden_channels": [16, 32],
    },
    "config": {"epochs": 4, "batch_size": 32, "learning_rate": 0.003, "seed": 42},
    "windowing": {"window_length": 30, "stride": 30},
}


@pytest.fixture
def client(model_data) -> Iterator[TestClient]:
    app = create_app()
    publisher = RecordingEventPublisher()
    data = DataContainer(repository=InMemoryDatasetRepository(), publisher=publisher)
    quality = DataQualityContainer(
        datasets=data.repository,
        assessments=InMemoryAssessmentRepository(),
        publisher=publisher,
    )
    model = ModelContainer(
        datasets=data.repository,
        assessments=quality.assessments,
        runs=InMemoryTrainingRunRepository(),
        publisher=publisher,
    )
    app.dependency_overrides[get_container] = lambda: data
    app.dependency_overrides[get_quality_container] = lambda: quality
    app.dependency_overrides[get_model_container] = lambda: model

    # 앞의 두 모듈은 Use Case 로 미리 통과시킨다. 여기서 볼 것은 모델 API 다.
    ms.pass_both_gates(
        data,
        quality,
        dataset_id="api-ds",
        assessment_id="api-qa",
        path=model_data.train,
    )
    with TestClient(app) as test_client:
        yield test_client


class TestPreparation:
    def test_준비하면_201_과_텐서_요약을_돌려준다(self, client) -> None:
        response = client.post("/training-runs", json=PREPARE_BODY)
        assert response.status_code == 201

        body = response.json()
        assert body["input_shape"] == [30, 6]
        assert body["batch_shape"] == [32, 30, 6]
        assert len(body["summaries"]) == 3
        assert body["summaries"][0]["sample_shape"] == [30, 6]

    def test_중복_준비는_409(self, client) -> None:
        client.post("/training-runs", json=PREPARE_BODY)
        assert client.post("/training-runs", json=PREPARE_BODY).status_code == 409

    def test_게이트를_통과하지_않은_데이터는_409(self, client, model_data) -> None:
        from tests.support import quality_scenario as qs

        container = client.app.dependency_overrides[get_container]()
        qs.prepare_dataset(container, "ungated", model_data.train)

        response = client.post(
            "/training-runs",
            json={**PREPARE_BODY, "run_id": "r2", "dataset_id": "ungated", "assessment_id": None},
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "ILLEGAL_STATE_TRANSITION"

    def test_창_길이와_입력_모양이_다르면_422(self, client) -> None:
        response = client.post(
            "/training-runs",
            json={
                **PREPARE_BODY,
                "run_id": "r3",
                "windowing": {"window_length": 60, "stride": 60},
            },
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "SHAPE_MISMATCH"

    def test_없는_학습_조회는_404(self, client) -> None:
        response = client.get("/training-runs/없음")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "TRAINING_RUN_NOT_FOUND"


class TestJobFlow:
    def _prepared(self, client) -> None:  # noqa: ANN001
        assert client.post("/training-runs", json=PREPARE_BODY).status_code == 201

    def test_시작하면_202_를_즉시_돌려준다(self, client) -> None:
        """학습이 끝나기를 기다리지 않는다. (CLAUDE.md §11)"""
        self._prepared(client)
        response = client.post("/training-runs/run-api/start")

        assert response.status_code == 202
        # 응답 시점의 상태는 아직 PREPARED 다 — 그래서 202(Accepted)다.
        assert response.json()["status"] == "PREPARED"

    def test_끝난_뒤_조회하면_COMPLETED(self, client) -> None:
        self._prepared(client)
        client.post("/training-runs/run-api/start")  # 백그라운드 작업이 여기서 끝난다

        body = client.get("/training-runs/run-api").json()
        assert body["status"] == "COMPLETED"
        assert body["epoch_count"] == 4
        assert body["model_version_id"] is not None

    def test_곡선을_따로_조회한다(self, client) -> None:
        self._prepared(client)
        client.post("/training-runs/run-api/start")

        body = client.get("/training-runs/run-api/curve").json()
        assert body["epoch_count"] == 4
        assert body["best_epoch"] is not None
        assert "epoch" in body["table"]

    def test_구조_프로파일을_조회한다(self, client) -> None:
        self._prepared(client)
        body = client.get("/training-runs/run-api/architecture").json()
        assert body["parameter_count"] > 0
        assert body["mac_count"] > 0

    def test_두_번_시작하면_두_번째는_실패한다(self, client) -> None:
        self._prepared(client)
        client.post("/training-runs/run-api/start")
        with pytest.raises(Exception):  # noqa: B017 - 백그라운드에서 터진다
            client.post("/training-runs/run-api/start")


class TestEvaluationAndAcceptance:
    @pytest.fixture
    def trained_client(self, client) -> TestClient:
        client.post("/training-runs", json=PREPARE_BODY)
        client.post("/training-runs/run-api/start")
        return client

    def test_평가는_혼동_행렬을_함께_돌려준다(self, trained_client) -> None:
        body = trained_client.post(
            "/training-runs/run-api/evaluations", json={"split": "test"}
        ).json()

        assert body["split"] == "test"
        assert 0.0 <= body["accuracy"] <= 1.0
        assert len(body["per_class"]) == 3
        assert body["latency_ms_p95"] > 0
        assert "실제 \\ 예측" in body["matrix"]

    def test_학습_전에는_평가할_수_없다(self, client) -> None:
        client.post("/training-runs", json=PREPARE_BODY)
        response = client.post(
            "/training-runs/run-api/evaluations", json={"split": "test"}
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "MODEL_NOT_TRAINED"

    def test_승인_판정(self, trained_client) -> None:
        trained_client.post(
            "/training-runs/run-api/evaluations", json={"split": "test"}
        )
        body = trained_client.post(
            "/training-runs/run-api/acceptance",
            json={
                "split": "test",
                "evaluation": {"critical_labels": ["FAULT"], "min_critical_recall": 0.9},
                "latency_p95_ms": 30.0,
            },
        ).json()

        assert body["verdict"] in ("PASSED", "PASSED_WITH_WARNINGS", "FAILED")
        assert body["model_version_id"].startswith("mv-")

    def test_지연시간_예산을_조이면_막힌다(self, trained_client) -> None:
        trained_client.post(
            "/training-runs/run-api/evaluations", json={"split": "test"}
        )
        body = trained_client.post(
            "/training-runs/run-api/acceptance",
            json={"split": "test", "latency_p95_ms": 0.001},
        ).json()

        assert body["is_deployable"] is False
        assert any(
            f["code"] == "ACCEPT_LATENCY_OVER_BUDGET" for f in body["blocking"]
        )

    def test_현장_홀드아웃_평가(self, trained_client, model_data) -> None:
        body = trained_client.post(
            "/training-runs/run-api/field-evaluations",
            json={"field_uri": str(model_data.field)},
        ).json()

        assert body["split"] == "field"
        assert body["accuracy"] > 0.0

    def test_판정_후_평가는_409_이고_reopen_으로_되돌린다(
        self, trained_client
    ) -> None:
        trained_client.post(
            "/training-runs/run-api/evaluations", json={"split": "test"}
        )
        trained_client.post("/training-runs/run-api/acceptance", json={})

        blocked = trained_client.post(
            "/training-runs/run-api/evaluations", json={"split": "validation"}
        )
        assert blocked.status_code == 409

        reopened = trained_client.post(
            "/training-runs/run-api/reopen", json={"reason": "현장 재수집분 반영"}
        )
        assert reopened.status_code == 200
        assert reopened.json()["status"] == "COMPLETED"


class TestOpenApi:
    def test_모델_엔드포인트가_문서에_노출된다(self, client) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        expected = {
            "/training-runs",
            "/training-runs/{run_id}",
            "/training-runs/{run_id}/start",
            "/training-runs/{run_id}/curve",
            "/training-runs/{run_id}/architecture",
            "/training-runs/{run_id}/evaluations",
            "/training-runs/{run_id}/field-evaluations",
            "/training-runs/{run_id}/acceptance",
            "/training-runs/{run_id}/reopen",
        }
        assert expected <= set(paths)
