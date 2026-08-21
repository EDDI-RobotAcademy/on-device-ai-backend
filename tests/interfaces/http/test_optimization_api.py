"""Optimization API — 계약과 오류 변환. (모듈 4)

여기서 확인하는 것은 세 가지다.
    1. Route 가 얇은가 (판단이 Domain 에 있는가)
    2. Domain 예외가 HTTP 상태로 제대로 바뀌는가 (CLAUDE.md §12)
    3. 오래 걸리는 작업이 요청을 붙잡지 않는가 (CLAUDE.md §11)
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from infrastructure.config.container import OptimizationContainer
from infrastructure.persistence.in_memory_optimization_run_repository import (
    InMemoryOptimizationRunRepository,
)
from interfaces.http.app import create_app
from interfaces.http.dependencies.container import get_optimization_container

BUDGET = {
    "name": "전력 감시 설비",
    "latency_p95_ms": 1.0,
    "storage_kib": 64.0,
    "activation_kib": 64.0,
    "min_macro_recall": 0.6,
    "max_accuracy_drop": 0.02,
    "max_class_recall_drop": 0.1,
}
FAST = {"warmup_runs": 5, "measured_runs": 30}
DEVICE = {
    "name": "edge-mcu",
    "peak_gmac_per_second": 2.0,
    "memory_bandwidth_gb_per_second": 1.6,
}


@pytest.fixture
def client(optimized, tmp_path) -> Iterator[TestClient]:  # noqa: ANN001
    """세션에서 학습·등록해 둔 모델을 그대로 쓰고, 저장소만 새로 만든다."""
    app = create_app()
    source = optimized.optimization
    container = OptimizationContainer(
        training_runs=source.training_runs,
        registry=source.registry,
        runs=InMemoryOptimizationRunRepository(),
        publisher=source.publisher,
        artifact_dir=tmp_path / "artifacts",
    )
    app.dependency_overrides[get_optimization_container] = lambda: container
    with TestClient(app) as test_client:
        test_client.training_run_id = optimized.training_run_id  # type: ignore[attr-defined]
        yield test_client


def start(client: TestClient, run_id: str = "opt-api", **overrides):  # noqa: ANN003, ANN201
    body = {
        "run_id": run_id,
        "training_run_id": client.training_run_id,  # type: ignore[attr-defined]
    }
    body.update(overrides)
    return client.post("/optimization-runs", json=body)


class Test시작:
    def test_시작하면_201_이다(self, client) -> None:
        response = start(client)
        assert response.status_code == 201

        body = response.json()
        assert body["status"] == "OPEN"
        assert body["candidate_labels"] == []

    def test_없는_학습은_404_다(self, client) -> None:
        response = client.post(
            "/optimization-runs",
            json={"run_id": "opt-x", "training_run_id": "없는-학습"},
        )
        assert response.status_code == 404

    def test_없는_최적화_조회는_404_다(self, client) -> None:
        assert client.get("/optimization-runs/없음").status_code == 404

    def test_요청_형식이_틀리면_422_다(self, client) -> None:
        response = client.post("/optimization-runs", json={"run_id": "opt-x"})
        assert response.status_code == 422


class Test측정과변환:
    def test_기준_측정은_프로토콜을_함께_돌려준다(self, client) -> None:
        start(client)
        response = client.post(
            "/optimization-runs/opt-api/baseline", json={"protocol": FAST}
        )
        assert response.status_code == 200

        body = response.json()
        assert "warmup=5" in body["protocol"]
        assert body["p95_ms"] >= body["p50_ms"]
        assert body["report"]

    def test_기준을_재기_전_변환은_409_다(self, client) -> None:
        start(client)
        response = client.post(
            "/optimization-runs/opt-api/candidates",
            json={"runtime": "ONNX", "protocol": FAST},
        )
        assert response.status_code == 409

    def test_변환하면_201_과_후보를_돌려준다(self, client) -> None:
        start(client)
        client.post("/optimization-runs/opt-api/baseline", json={"protocol": FAST})
        response = client.post(
            "/optimization-runs/opt-api/candidates",
            json={"runtime": "ONNX", "protocol": FAST, "equivalence_samples": 40},
        )
        assert response.status_code == 201

        body = response.json()
        assert body["label"] == "ONNX/FP32"
        assert body["theoretical_weight_bytes"] + body["overhead_bytes"] == body["size_bytes"]
        assert "argmax 일치" in body["equivalence"]

    def test_지원하지_않는_조합은_409_다(self, client) -> None:
        """요청 형식은 맞다. 이 시스템이 그 조합을 만들 수 없을 뿐이다."""
        start(client)
        client.post("/optimization-runs/opt-api/baseline", json={"protocol": FAST})
        response = client.post(
            "/optimization-runs/opt-api/candidates",
            json={"runtime": "ONNX", "precision": "INT8", "protocol": FAST},
        )
        assert response.status_code == 409
        assert "어댑터가 없다" in response.json()["error"]["message"]

    def test_모르는_런타임은_요청_단계에서_막힌다(self, client) -> None:
        start(client)
        response = client.post(
            "/optimization-runs/opt-api/candidates", json={"runtime": "TENSORRT"}
        )
        assert response.status_code == 422

    def test_비동기_변환은_202_로_즉시_돌아온다(self, client) -> None:
        """CLAUDE.md §11 — 요청이 변환을 붙잡고 기다리지 않는다."""
        start(client)
        client.post("/optimization-runs/opt-api/baseline", json={"protocol": FAST})
        response = client.post(
            "/optimization-runs/opt-api/candidates:async",
            json={"runtime": "TORCHSCRIPT", "protocol": FAST},
        )
        assert response.status_code == 202

        # 배경 작업이 끝난 뒤 상태를 물어본다.
        body = client.get("/optimization-runs/opt-api").json()
        assert "TORCHSCRIPT/FP32" in body["candidate_labels"]


class Test분석과선택:
    @pytest.fixture
    def prepared(self, client) -> TestClient:
        start(client)
        client.post("/optimization-runs/opt-api/baseline", json={"protocol": FAST})
        for runtime in ("ONNX", "TFLITE"):
            client.post(
                "/optimization-runs/opt-api/candidates",
                json={"runtime": runtime, "protocol": FAST, "equivalence_samples": 40},
            )
        return client

    def test_비교표를_돌려준다(self, prepared) -> None:
        body = prepared.get("/optimization-runs/opt-api/tradeoff").json()
        assert "ONNX/FP32" in body["report"]
        assert body["pareto_front"]

    def test_크기_분해를_돌려준다(self, prepared) -> None:
        rows = prepared.get("/optimization-runs/opt-api/artifacts").json()
        assert len(rows) == 3  # 기준 + 후보 둘
        for row in rows:
            assert row["overhead_bytes"] >= 0

    def test_병목_분석은_저장되고_다시_조회된다(self, prepared) -> None:
        posted = prepared.post(
            "/optimization-runs/opt-api/roofline", json={"device": DEVICE}
        )
        assert posted.status_code == 200
        assert posted.json()["machine_balance"] == pytest.approx(1.25)

        fetched = prepared.get("/optimization-runs/opt-api/roofline").json()
        assert fetched["total_macs"] == posted.json()["total_macs"]

    def test_분석하기_전_조회는_404_다(self, client) -> None:
        start(client)
        assert client.get("/optimization-runs/opt-api/roofline").status_code == 404

    def test_선택은_판정과_근거를_함께_돌려준다(self, prepared) -> None:
        response = prepared.post(
            "/optimization-runs/opt-api/selection", json={"budget": BUDGET}
        )
        assert response.status_code == 200

        body = response.json()
        assert body["verdict"] == "PASSED"
        assert body["selected_label"]
        assert any(r["label"] == "PYTORCH/FP32" for r in body["rejected"])

    def test_판정_뒤_변환은_409_다(self, prepared) -> None:
        prepared.post("/optimization-runs/opt-api/selection", json={"budget": BUDGET})
        response = prepared.post(
            "/optimization-runs/opt-api/candidates",
            json={"runtime": "TORCHSCRIPT", "protocol": FAST},
        )
        assert response.status_code == 409

    def test_이유를_주면_되돌릴_수_있다(self, prepared) -> None:
        prepared.post("/optimization-runs/opt-api/selection", json={"budget": BUDGET})
        response = prepared.post(
            "/optimization-runs/opt-api/reopen", json={"reason": "예산이 바뀌었다"}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "EXPLORING"

    def test_빈_이유는_422_다(self, prepared) -> None:
        prepared.post("/optimization-runs/opt-api/selection", json={"budget": BUDGET})
        response = prepared.post("/optimization-runs/opt-api/reopen", json={"reason": ""})
        assert response.status_code == 422

    def test_후보가_없으면_선택은_409_다(self, client) -> None:
        start(client)
        client.post("/optimization-runs/opt-api/baseline", json={"protocol": FAST})
        response = client.post(
            "/optimization-runs/opt-api/selection", json={"budget": BUDGET}
        )
        assert response.status_code == 409

    def test_판정_전_판정_조회는_404_다(self, prepared) -> None:
        assert prepared.get("/optimization-runs/opt-api/selection").status_code == 404

    def test_목록에_나온다(self, prepared) -> None:
        rows = prepared.get("/optimization-runs").json()
        assert [r["run_id"] for r in rows] == ["opt-api"]
