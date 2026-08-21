"""Operations API — 계약과 오류 변환. (모듈 5)

두 자원으로 나뉘어 있다. **Aggregate 가 둘이기 때문이다.**

    /deployments      결정
    /health-watches   사실
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from infrastructure.config.container import OperationsContainer
from infrastructure.persistence.in_memory_deployment_repository import (
    InMemoryDeploymentRepository,
    InMemoryHealthWatchRepository,
)
from interfaces.http.app import create_app
from interfaces.http.dependencies.container import get_operations_container

DEPLOYMENT_ID = "dep-line3"
"""세션 파이프라인과 **같은 ID** 다.

추론 로그는 배포 ID 로 묶여 있다. 다른 ID 로 배포하면
34,533건이 그대로 있는데도 이 배포에는 로그가 하나도 없다 —
그것 자체가 실습 5-3 이 말하는 상황이다.
"""

TARGET = {
    "kind": "DEVICE_GROUP",
    "identifier": "LINE-3",
    "name": "3라인 전력 감시",
    "device_count": 3,
}
LATENCY_POLICY = {"cycle_budget_ms": 30.0, "max_regression_ratio": 20.0}


@pytest.fixture
def client(deployed) -> Iterator[TestClient]:  # noqa: ANN001
    """세션에서 만든 로그와 측정기를 그대로 쓰고, 저장소만 새로 만든다."""
    app = create_app()
    source = deployed.operations
    container = OperationsContainer(
        optimization_runs=source.optimization_runs,
        training_runs=source.training_runs,
        deployments=InMemoryDeploymentRepository(),
        watches=InMemoryHealthWatchRepository(),
        logs=source.logs,
        publisher=source.publisher,
        drift=source.drift,
    )
    app.dependency_overrides[get_operations_container] = lambda: container
    with TestClient(app) as test_client:
        test_client.optimization_run_id = deployed.optimized.run_id  # type: ignore[attr-defined]
        test_client.training_run_id = deployed.trained.run_id  # type: ignore[attr-defined]
        yield test_client


def deploy(client: TestClient, **overrides):  # noqa: ANN003, ANN201
    body: dict[str, object] = {
        "deployment_id": DEPLOYMENT_ID,
        "optimization_run_id": client.optimization_run_id,  # type: ignore[attr-defined]
        "training_run_id": client.training_run_id,  # type: ignore[attr-defined]
        "target": TARGET,
        "watch_id": "watch-api",
        "released_at": "2026-05-19 23:00:00",
    }
    body.update(overrides)
    return client.post("/deployments", json=body)


def window(day: int = 20, hour: int = 0, **overrides) -> dict[str, object]:  # noqa: ANN003
    body: dict[str, object] = {
        "label": f"05-{day} {hour:02d}시",
        "started_at": f"2026-05-{day} {hour:02d}:00:00",
        "ended_at": f"2026-05-{day} {hour + 7:02d}:59:59",
    }
    body.update(overrides)
    return body


class Test배포:
    def test_배포하면_201_과_관측이_함께_열린다(self, client) -> None:
        response = deploy(client)
        assert response.status_code == 201

        body = response.json()
        assert body["deployment"]["status"] == "DEPLOYED"
        assert body["deployment"]["current_version"] == 1
        assert body["watch_id"] == "watch-api"
        assert body["check"]["verdict"] in ("PASSED", "PASSED_WITH_WARNINGS")

    def test_첫_배포를_너무_넓게_하면_점검에_걸린다(self, client) -> None:
        response = deploy(
            client,
            target={"kind": "FLEET", "identifier": "ALL", "device_count": 1200},
        )
        assert response.status_code == 201  # 배포는 된다
        body = response.json()
        assert not body["check"]["can_release"]
        codes = {f["code"] for f in body["check"]["findings"]}
        assert "RELEASE_FIRST_TOO_WIDE" in codes

    def test_없는_최적화는_404_다(self, client) -> None:
        assert deploy(client, optimization_run_id="없음").status_code == 404

    def test_요청_형식이_틀리면_422_다(self, client) -> None:
        assert client.post("/deployments", json={"deployment_id": "x"}).status_code == 422

    def test_없는_배포_조회는_404_다(self, client) -> None:
        assert client.get("/deployments/없음").status_code == 404

    def test_목록에_나온다(self, client) -> None:
        deploy(client)
        rows = client.get("/deployments").json()
        assert [r["deployment_id"] for r in rows] == [DEPLOYMENT_ID]


class Test버전과롤백:
    @pytest.fixture
    def three(self, client) -> TestClient:
        deploy(client)
        for label, moment in (
            ("TFLITE/INT8", "2026-05-21 12:00:00"),
            ("ONNX/FP32", "2026-05-22 12:00:00"),
        ):
            client.post(
                f"/deployments/{DEPLOYMENT_ID}/versions",
                json={
                    "optimization_run_id": client.optimization_run_id,  # type: ignore[attr-defined]
                    "training_run_id": client.training_run_id,  # type: ignore[attr-defined]
                    "artifact_label": label,
                    "released_at": moment,
                    "require_selected": False,
                },
            )
        return client

    def test_새_버전은_201_이다(self, three) -> None:
        body = three.get(f"/deployments/{DEPLOYMENT_ID}").json()
        assert body["current_version"] == 3
        assert body["version_count"] == 3

    def test_롤백은_새_버전을_만든다(self, three) -> None:
        response = three.post(
            f"/deployments/{DEPLOYMENT_ID}/rollback",
            json={
                "to_version": 1,
                "reason": "v3 배포 후 FAULT 재현율 붕괴",
                "occurred_at": "2026-05-23 09:00:00",
            },
        )
        assert response.status_code == 200

        body = response.json()
        assert body["current_version"] == 4
        assert body["rollback_count"] == 1
        assert body["status"] == "ROLLED_BACK"

    def test_없는_버전으로는_404_다(self, three) -> None:
        response = three.post(
            f"/deployments/{DEPLOYMENT_ID}/rollback",
            json={"to_version": 9, "reason": "없는 버전"},
        )
        assert response.status_code == 404

    def test_이유_없는_롤백은_422_다(self, three) -> None:
        response = three.post(
            f"/deployments/{DEPLOYMENT_ID}/rollback", json={"to_version": 1, "reason": ""}
        )
        assert response.status_code == 422

    def test_현재_버전으로_롤백은_409_다(self, three) -> None:
        response = three.post(
            f"/deployments/{DEPLOYMENT_ID}/rollback",
            json={"to_version": 3, "reason": "현재 버전"},
        )
        assert response.status_code == 409


class Test관측:
    @pytest.fixture
    def observing(self, client) -> TestClient:
        deploy(client)
        return client

    def test_관측하면_판정과_근거가_함께_온다(self, observing) -> None:
        response = observing.post(
            f"/deployments/{DEPLOYMENT_ID}/observations",
            json={"window": window(23, 0), "latency_policy": LATENCY_POLICY},
        )
        assert response.status_code == 200

        body = response.json()
        assert body["verdict"] == "FAILED"
        assert body["sample_count"] > 1000
        assert body["max_psi"] > 5.0
        assert body["quarantine_recommended"]
        assert body["findings"]

    def test_시간선을_돌려준다(self, observing) -> None:
        for day in (20, 21, 22, 23):
            observing.post(
                f"/deployments/{DEPLOYMENT_ID}/observations",
                json={"window": window(day, 0), "latency_policy": LATENCY_POLICY},
            )
        body = observing.get("/health-watches/watch-api/timeline").json()
        assert body["window_count"] == 4
        assert body["report"]

    def test_언제부터인지_찾는다(self, observing) -> None:
        for day in (20, 21, 22, 23):
            for hour in (0, 8, 16):
                observing.post(
                    f"/deployments/{DEPLOYMENT_ID}/observations",
                    json={"window": window(day, hour), "latency_policy": LATENCY_POLICY},
                )
        body = observing.post(
            "/health-watches/watch-api/onset",
            json={"metric": "INPUT_PSI", "threshold": 0.2, "consecutive": 3},
        ).json()

        assert body["is_sustained"]
        assert body["first_exceeded"] != body["sustained_from"]

    def test_모르는_지표는_422_다(self, observing) -> None:
        response = observing.post(
            "/health-watches/watch-api/onset", json={"metric": "무엇이든"}
        )
        assert response.status_code == 422

    def test_없는_관측은_404_다(self, client) -> None:
        assert client.get("/health-watches/없음/timeline").status_code == 404

    def test_기준_재고정은_이유가_필수다(self, observing) -> None:
        empty = observing.post(
            f"/deployments/{DEPLOYMENT_ID}/baseline",
            json={"window": window(20, 0), "reason": ""},
        )
        assert empty.status_code == 422

        response = observing.post(
            f"/deployments/{DEPLOYMENT_ID}/baseline",
            json={"window": window(20, 0), "reason": "배포 직후 안정 구간"},
        )
        assert response.status_code == 200
        assert sum(response.json()["baseline_mix"].values()) == pytest.approx(1.0)

    def test_관측_없이_배포하면_관측_조회가_404_다(self, client) -> None:
        deploy(client)
        client_container_watches_cleared(client)
        assert client.get(f"/deployments/{DEPLOYMENT_ID}/watch").status_code == 404


class Test격리:
    def test_근거가_있으면_이유_없이도_멈춘다(self, client) -> None:
        deploy(client)
        client.post(
            f"/deployments/{DEPLOYMENT_ID}/observations",
            json={"window": window(23, 0), "latency_policy": LATENCY_POLICY},
        )
        response = client.post(f"/deployments/{DEPLOYMENT_ID}/quarantine", json={})
        assert response.status_code == 200
        assert response.json()["status"] == "QUARANTINED"
        assert response.json()["quarantine_reason"]

    def test_근거가_없으면_409_다(self, client) -> None:
        deploy(client)
        client.post(
            f"/deployments/{DEPLOYMENT_ID}/observations",
            json={"window": window(20, 0), "latency_policy": LATENCY_POLICY},
        )
        response = client.post(f"/deployments/{DEPLOYMENT_ID}/quarantine", json={})
        assert response.status_code == 409

    def test_격리_중_새_버전은_409_다(self, client) -> None:
        deploy(client)
        client.post(
            f"/deployments/{DEPLOYMENT_ID}/observations",
            json={"window": window(23, 0), "latency_policy": LATENCY_POLICY},
        )
        client.post(f"/deployments/{DEPLOYMENT_ID}/quarantine", json={})
        response = client.post(
            f"/deployments/{DEPLOYMENT_ID}/versions",
            json={
                "optimization_run_id": client.optimization_run_id,  # type: ignore[attr-defined]
                "require_selected": False,
            },
        )
        assert response.status_code == 409

    def test_해제에는_이유가_필요하다(self, client) -> None:
        deploy(client)
        client.post(
            f"/deployments/{DEPLOYMENT_ID}/observations",
            json={"window": window(23, 0), "latency_policy": LATENCY_POLICY},
        )
        client.post(f"/deployments/{DEPLOYMENT_ID}/quarantine", json={})

        assert (
            client.post(f"/deployments/{DEPLOYMENT_ID}/resume", json={"reason": ""}).status_code
            == 422
        )
        response = client.post(
            f"/deployments/{DEPLOYMENT_ID}/resume", json={"reason": "팬 교체 완료"}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "DEPLOYED"

    def test_사건을_조회하고_닫는다(self, client) -> None:
        deploy(client)
        client.post(
            f"/deployments/{DEPLOYMENT_ID}/observations",
            json={"window": window(23, 0), "latency_policy": LATENCY_POLICY},
        )
        incidents = client.get(
            "/health-watches/watch-api/incidents", params={"only_open": True}
        ).json()
        assert incidents

        incident_id = incidents[0]["incident_id"]
        assert (
            client.post(
                f"/health-watches/watch-api/incidents/{incident_id}/resolution",
                json={"resolution": ""},
            ).status_code
            == 422
        )
        response = client.post(
            f"/health-watches/watch-api/incidents/{incident_id}/resolution",
            json={"resolution": "여름 데이터로 재학습 후 v2 배포"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "RESOLVED"


class Test재학습:
    def test_판정과_막고_있는_것을_함께_돌려준다(self, client) -> None:
        deploy(client)
        for day in (20, 21, 22, 23):
            for hour in (0, 8, 16):
                client.post(
                    f"/deployments/{DEPLOYMENT_ID}/observations",
                    json={"window": window(day, hour), "latency_policy": LATENCY_POLICY},
                )
        response = client.post(
            "/health-watches/watch-api/retraining",
            json={"policy": {"min_labels_per_class": 100}},
        )
        assert response.status_code == 200

        body = response.json()
        assert body["needed"]
        assert "INPUT_DRIFT" in body["reasons"]
        assert body["blockers"]
        assert not body["can_start"]

    def test_관측이_없으면_404_다(self, client) -> None:
        assert (
            client.post("/health-watches/없음/retraining", json={}).status_code == 404
        )

    def test_그림자_실행기가_없으면_409_다(self, client) -> None:
        deploy(client)
        response = client.post(
            f"/deployments/{DEPLOYMENT_ID}/shadow",
            json={"window": window(23, 0), "candidate_artifact_id": "무엇이든"},
        )
        assert response.status_code == 409


def client_container_watches_cleared(client: TestClient) -> None:
    """관측 저장소만 비워 '배포는 됐는데 아무도 안 보고 있다' 를 만든다."""
    from interfaces.http.dependencies.container import get_operations_container

    override = client.app.dependency_overrides[get_operations_container]
    override().watches.clear()
