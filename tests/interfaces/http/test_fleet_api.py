"""Fleet API — 계약과 오류 변환. (모듈 6)

**API 에도 AWS 가 없다.** 클라이언트는 데이터가 S3 로 가는지 GCS 로 가는지 모른다.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from interfaces.http.app import create_app
from interfaces.http.dependencies.container import get_fleet_container
from tests.support import fleet_scenario as fs

FLEET_ID = "line3"


def device_body(index: int) -> dict[str, object]:
    group = "pilot" if index < 2 else "line-a" if index < 10 else "line-b"
    return {
        "device_id": f"DEV-{index:02d}",
        "group": group,
        "current_version": "v1.0.0",
        "last_seen_at": "2026-05-23 09:00:00",
    }


BUNDLE = {
    "release_id": "rel-v2",
    "version": "v2.0.0",
    "model_version_id": "mv-1",
    "artifact_uri": "s3://ondevice-ai-artifacts/v2/model.tflite",
    "artifact_bytes": 11_724,
    "checksum": "d" * 32,
    "runtime": "TFLITE",
    "precision": "FP16",
    "class_labels": ["FAULT", "OVERLOAD", "NORMAL"],
    "input_fields": ["active_power_kw"],
    "normalization": {"active_power_kw": [147.8, 39.8]},
    "expected_p95_ms": 0.0031,
    "expected_class_mix": {"NORMAL": 0.78},
    "sample_interval_seconds": 10,
    "window_length": 30,
    "source_build_id": "build-1",
    "source_job_id": "train-1",
}


@pytest.fixture
def client(fleet_bare) -> Iterator[TestClient]:  # noqa: ANN001
    app = create_app()
    app.dependency_overrides[get_fleet_container] = lambda: fleet_bare
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def seeded(client) -> TestClient:  # noqa: ANN001
    client.post(
        "/fleets",
        json={
            "fleet_id": FLEET_ID,
            "name": "3라인 전력 감시",
            "devices": [device_body(n) for n in range(24)],
        },
    )
    return client


class Test플릿:
    def test_만들면_201_이고_집계를_돌려준다(self, client) -> None:
        response = client.post(
            "/fleets",
            json={
                "fleet_id": FLEET_ID,
                "name": "3라인",
                "devices": [device_body(n) for n in range(4)],
            },
        )
        assert response.status_code == 201

        body = response.json()
        assert body["size"] == 4
        assert body["report"]

    def test_없는_플릿은_404_다(self, client) -> None:
        assert client.get("/fleets/없음").status_code == 404

    def test_요청_형식이_틀리면_422_다(self, client) -> None:
        assert client.post("/fleets", json={"fleet_id": "x"}).status_code == 422

    def test_집계에_소견이_붙는다(self, seeded) -> None:
        body = seeded.get(f"/fleets/{FLEET_ID}").json()
        assert body["version_count"] == 1
        assert body["dominant_version"] == "v1.0.0"


class Test업링크:
    def body(self, **overrides) -> dict[str, object]:  # noqa: ANN003
        base: dict[str, object] = {
            "batch": {
                "device_id": "DEV-02",
                "kind": "INFERENCE_LOG",
                "window_start": "2026-05-23 09:00:00",
                "window_end": "2026-05-23 09:59:59",
                "record_count": 360,
                "payload_bytes": 7_200,
                "checksum": "c0ffee",
                "fields": ["occurred_at", "predicted_label"],
            },
            "body_base64": base64.b64encode(b'{"x":1}' * 900).decode(),
        }
        base.update(overrides)
        return base

    def test_올리면_202_다(self, seeded) -> None:
        response = seeded.post(f"/fleets/{FLEET_ID}/uplinks", json=self.body())
        assert response.status_code == 202

        body = response.json()
        assert body["accepted"]
        assert body["uri"].startswith("s3://")

    def test_개인정보가_섞이면_저장하지_않는다(self, seeded) -> None:
        batch = self.body()["batch"]
        batch["fields"] = ["occurred_at", "operator_name"]  # type: ignore[index]
        response = seeded.post(
            f"/fleets/{FLEET_ID}/uplinks", json=self.body(batch=batch)
        )
        assert response.status_code == 202

        body = response.json()
        assert not body["accepted"]
        assert body["uri"] == ""
        assert "UPLINK_FORBIDDEN_FIELD" in {f["code"] for f in body["findings"]}

    def test_등록되지_않은_디바이스는_404_다(self, seeded) -> None:
        batch = self.body()["batch"]
        batch["device_id"] = "DEV-999"  # type: ignore[index]
        assert (
            seeded.post(
                f"/fleets/{FLEET_ID}/uplinks", json=self.body(batch=batch)
            ).status_code
            == 404
        )


class Test데이터레이크:
    def test_좁힌_접두어와_소견을_돌려준다(self, seeded) -> None:
        for part in range(3):
            seeded.post(
                f"/fleets/{FLEET_ID}/uplinks",
                json={
                    "batch": {
                        "device_id": "DEV-02",
                        "window_start": "2026-05-23 09:00:00",
                        "window_end": "2026-05-23 09:59:59",
                        "record_count": 360,
                        "payload_bytes": 7_200,
                        "checksum": "x",
                    },
                    "body_base64": base64.b64encode(b"x" * 5_000).decode(),
                    "part": part,
                },
            )
        body = seeded.post(
            f"/fleets/{FLEET_ID}/lake",
            json={
                "filters": {
                    "kind": "inference_log",
                    "device": "DEV-02",
                    "date": "2026-05-23",
                }
            },
        ).json()

        assert body["object_count"] == 3
        assert body["can_narrow"]
        assert "device=DEV-02" in body["narrowed_prefix"]


class Test데이터셋과학습:
    def dataset_body(self) -> dict[str, object]:
        return {
            "build_id": "build-1",
            "window": {
                "started_at": "2026-05-22 00:00:00",
                "ended_at": "2026-05-23 23:59:59",
                "reason": "드리프트 이후",
            },
            "record_counts": {f"DEV-{n:02d}": 900 for n in range(24)},
            "labeled_counts": {f"DEV-{n:02d}": 110 for n in range(24)},
            "label_distribution": {"NORMAL": 1900, "OVERLOAD": 600, "FAULT": 140},
        }

    def test_데이터셋을_만들면_201_이다(self, seeded) -> None:
        response = seeded.post(f"/fleets/{FLEET_ID}/datasets", json=self.dataset_body())
        assert response.status_code == 201

        body = response.json()
        assert body["can_build"]
        assert body["dataset_uri"].startswith("s3://")

    def test_격리된_디바이스는_빠지고_이유가_남는다(self, seeded) -> None:
        seeded.post(
            f"/fleets/{FLEET_ID}/devices/DEV-05/status",
            json={"status": "QUARANTINED", "note": "모듈 5 격리"},
        )
        body = seeded.post(
            f"/fleets/{FLEET_ID}/datasets", json=self.dataset_body()
        ).json()

        excluded = {row["device_id"] for row in body["excluded"]}
        assert "DEV-05" in excluded
        assert all(row["reason"] for row in body["excluded"])

    def test_학습_제출은_202_다(self, seeded) -> None:
        response = seeded.post(
            "/cloud-training-jobs",
            json={
                "job_id": "train-1",
                "dataset_uri": "s3://ondevice-ai-lake/datasets/build=build-1/",
                "output_uri": "s3://ondevice-ai-artifacts/train-1/",
                "compute": {
                    "instance_type": "ml.m5.large",
                    "max_runtime_seconds": 3600,
                    "hourly_cost_usd": 0.13,
                },
            },
        )
        assert response.status_code == 202
        assert response.json()["job_id"] == "train-1"

    def test_예산을_넘으면_제출하지_않는다(self, seeded) -> None:
        body = seeded.post(
            "/cloud-training-jobs",
            json={
                "job_id": "train-expensive",
                "dataset_uri": "s3://a/",
                "output_uri": "s3://b/",
                "compute": {
                    "instance_type": "ml.p3.8xlarge",
                    "instance_count": 4,
                    "max_runtime_seconds": 28_800,
                    "hourly_cost_usd": 17.6,
                },
            },
        ).json()
        assert "TRAIN_OVER_BUDGET" in {f["code"] for f in body["findings"]}
        assert body["status"] == "PENDING"

    def test_상태를_물어본다(self, seeded) -> None:
        seeded.post(
            "/cloud-training-jobs",
            json={
                "job_id": "train-2",
                "dataset_uri": "s3://a/",
                "output_uri": "s3://b/",
                "compute": {"instance_type": "ml.m5.large"},
            },
        )
        body = seeded.get("/cloud-training-jobs/train-2").json()
        assert body["job_id"] == "train-2"

    def test_없는_학습은_502_다(self, seeded) -> None:
        """인프라가 못 찾는 것은 인프라 오류다."""
        assert seeded.get("/cloud-training-jobs/없음").status_code == 502


class Test릴리스와채널:
    def test_릴리스를_등록하면_201_이다(self, seeded) -> None:
        response = seeded.post(f"/fleets/{FLEET_ID}/releases", json={"bundle": BUNDLE})
        assert response.status_code == 201
        assert response.json()["can_publish"]

    def test_전처리가_빠지면_점검에_걸린다(self, seeded) -> None:
        bundle = dict(BUNDLE, version="v2.0.1", normalization={})
        body = seeded.post(
            f"/fleets/{FLEET_ID}/releases", json={"bundle": bundle}
        ).json()
        assert not body["can_publish"]
        assert "RELEASE_NO_PREPROCESSING" in {f["code"] for f in body["findings"]}

    def test_canary를_거쳐야_stable로_간다(self, seeded) -> None:
        seeded.post(f"/fleets/{FLEET_ID}/releases", json={"bundle": BUNDLE})
        assert (
            seeded.post(
                f"/fleets/{FLEET_ID}/channels",
                json={"version": "v2.0.0", "channel": "STABLE"},
            ).status_code
            == 409
        )
        assert (
            seeded.post(
                f"/fleets/{FLEET_ID}/channels",
                json={"version": "v2.0.0", "channel": "CANARY"},
            ).status_code
            == 200
        )

    def test_등록되지_않은_버전은_409_다(self, seeded) -> None:
        assert (
            seeded.post(
                f"/fleets/{FLEET_ID}/channels",
                json={"version": "v9.9.9", "channel": "CANARY"},
            ).status_code
            == 409
        )


class Test롤아웃:
    @pytest.fixture
    def rolling(self, seeded) -> TestClient:  # noqa: ANN001
        seeded.post(f"/fleets/{FLEET_ID}/releases", json={"bundle": BUNDLE})
        seeded.post(
            f"/fleets/{FLEET_ID}/channels",
            json={"version": "v2.0.0", "channel": "CANARY"},
        )
        seeded.post(
            f"/fleets/{FLEET_ID}/rollouts",
            json={
                "rollout_id": "ro-1",
                "version": "v2.0.0",
                "wave_sizes": [2, 8, 9999],
                "group_order": ["pilot", "line-a", "line-b"],
                "occurred_at": "2026-05-24 12:00:00",
            },
        )
        return seeded

    def test_계획은_202_다(self, seeded) -> None:
        seeded.post(f"/fleets/{FLEET_ID}/releases", json={"bundle": BUNDLE})
        response = seeded.post(
            f"/fleets/{FLEET_ID}/rollouts",
            json={"rollout_id": "ro-1", "version": "v2.0.0"},
        )
        assert response.status_code == 202
        assert response.json()["status"] == "RUNNING"

    def test_등록되지_않은_버전은_404_다(self, seeded) -> None:
        assert (
            seeded.post(
                f"/fleets/{FLEET_ID}/rollouts",
                json={"rollout_id": "ro-x", "version": "v9.9.9"},
            ).status_code
            == 404
        )

    def test_첫_단계가_너무_크면_422_다(self, seeded) -> None:
        seeded.post(f"/fleets/{FLEET_ID}/releases", json={"bundle": BUNDLE})
        assert (
            seeded.post(
                f"/fleets/{FLEET_ID}/rollouts",
                json={
                    "rollout_id": "ro-wide",
                    "version": "v2.0.0",
                    "wave_sizes": [15, 9999],
                },
            ).status_code
            == 422
        )

    def test_결과를_걷고_다음_단계로_간다(self, rolling) -> None:
        wave = rolling.post("/rollouts/ro-1/collect", json={}).json()
        assert wave["wave"] == "wave-1"

        advanced = rolling.post(
            "/rollouts/ro-1/advance", json={"fleet_id": FLEET_ID}
        )
        assert advanced.status_code == 200
        assert advanced.json()["current_wave"] == "wave-2"

    def test_없는_롤아웃은_404_다(self, client) -> None:
        assert client.get("/rollouts/없음").status_code == 404

    def test_이유_없는_중단은_422_다(self, rolling) -> None:
        assert (
            rolling.post("/rollouts/ro-1/halt", json={"reason": ""}).status_code == 422
        )

    def test_되돌리기는_202_다(self, rolling) -> None:
        rolling.post("/rollouts/ro-1/collect", json={})
        rolling.post("/rollouts/ro-1/apply", json={
            "fleet_id": FLEET_ID, "seen_at": "2026-05-24 13:00:00"
        })
        rolling.post(f"/fleets/{FLEET_ID}/releases", json=dict(
            bundle=dict(BUNDLE, version="v1.0.0", release_id="rel-v1")
        ))

        response = rolling.post(
            "/rollouts/ro-1/rollback",
            json={
                "fleet_id": FLEET_ID,
                "new_rollout_id": "ro-1-rb",
                "reason": "FAULT 재현율 붕괴",
                "to_version": "v1.0.0",
                "occurred_at": "2026-05-25 09:00:00",
            },
        )
        assert response.status_code == 202

        body = response.json()
        assert body["rollout_id"] == "ro-1-rb"
        assert body["coverage"] == 0.0

    def test_목록에_나온다(self, rolling) -> None:
        rows = rolling.get("/rollouts").json()
        assert [r["rollout_id"] for r in rows] == ["ro-1"]


class Test계보:
    def test_사슬을_돌려준다(self, seeded) -> None:
        seeded.post(f"/fleets/{FLEET_ID}/releases", json={"bundle": BUNDLE})
        body = seeded.post(
            f"/fleets/{FLEET_ID}/devices/DEV-00/lineage",
            json={"source_devices": ["DEV-00"], "window": "2026-05-22 ~ 2026-05-23"},
        ).json()

        assert body["device_id"] == "DEV-00"
        assert body["report"]

    def test_없는_디바이스는_404_다(self, seeded) -> None:
        assert (
            seeded.post(
                f"/fleets/{FLEET_ID}/devices/DEV-999/lineage", json={}
            ).status_code
            == 404
        )


class TestAWS없이:
    def test_어댑터가_없으면_409_다(self) -> None:
        """**자격증명 없이도 서버는 뜬다.** 쓰려고 할 때 막힌다."""
        from infrastructure.config.container import FleetContainer

        app = create_app()
        app.dependency_overrides[get_fleet_container] = lambda: FleetContainer()
        with TestClient(app) as bare:
            bare.post("/fleets", json={"fleet_id": "f", "name": "n", "devices": []})
            response = bare.post(
                "/fleets/f/uplinks",
                json={
                    "batch": {
                        "device_id": "D",
                        "window_start": "2026-05-23 09:00:00",
                        "window_end": "2026-05-23 09:59:59",
                        "record_count": 1,
                        "payload_bytes": 1,
                    }
                },
            )
        assert response.status_code == 409
