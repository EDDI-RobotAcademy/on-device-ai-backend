"""디바이스에 내려온 묶음을 읽고 **검증한다.**

`ReleaseBundle`(실습 6-6)이 회선을 건너오면 디스크에는 이렇게 남는다.

    bundles/v1.3.0/
      manifest.json   버전 · 체크섬 · 계약 · 알람 규칙
      model.tflite    결과물

읽기 전에 두 가지를 반드시 확인한다.

    체크섬   좁은 회선에서 **절반만 온 파일도 파일이다.** 열리고 파싱도 된다.
    계약     전처리와 라벨 순서. 어긋나면 아무 에러 없이 틀린다 (실습 5-12).

이 검증을 건너뛰면 그 대가는 현장이 낸다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from domain.operations.alerting import AlertRule
from domain.operations.pipeline import PipelineContract

MANIFEST = "manifest.json"
MODEL = "model.tflite"


class BundleRejected(RuntimeError):
    """묶음을 받아들일 수 없다. **받아들이지 않는 것이 옳은 상황이다.**"""


@dataclass(frozen=True, slots=True)
class DeployedBundle:
    """디스크에 놓여 있고 검증까지 끝난 묶음."""

    root: Path
    version: str
    model_version_id: str
    checksum: str
    contract: PipelineContract
    alert_rule: AlertRule
    expected_p95_ms: float = 0.0

    @property
    def model_path(self) -> Path:
        return self.root / MODEL

    def describe(self) -> str:
        return (
            f"{self.version} ({self.model_version_id}) "
            f"{self.contract.describe()} · {self.alert_rule.describe()}"
        )


def checksum_of(path: Path, *, chunk: int = 1 << 16) -> str:
    """파일 전체를 조각내어 읽는다. **보드 RAM 에 통째로 올리지 않는다.**"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def load_bundle(root: Path) -> DeployedBundle:
    """묶음을 읽고 검증한다. 하나라도 안 맞으면 **거절한다.**"""
    manifest_path = root / MANIFEST
    model_path = root / MODEL
    for path in (manifest_path, model_path):
        if not path.is_file():
            raise BundleRejected(f"묶음이 완전하지 않다: {path.name} 이 없다")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleRejected(f"manifest 를 읽을 수 없다: {exc}") from exc

    declared = str(manifest.get("checksum", "")).strip()
    if not declared:
        raise BundleRejected(
            "체크섬이 없다. **잘려 도착한 파일을 알아챌 방법이 없다** (실습 6-1)"
        )
    actual = checksum_of(model_path)
    if actual != declared:
        raise BundleRejected(
            f"체크섬이 다르다 (선언 {declared[:12]}… / 실제 {actual[:12]}…). "
            "전송 중 잘렸거나 다른 파일이다 — **절반만 온 파일도 열린다.**"
        )

    size = int(manifest.get("artifact_bytes", 0))
    if size and model_path.stat().st_size != size:
        raise BundleRejected(
            f"크기가 다르다 (선언 {size:,}B / 실제 {model_path.stat().st_size:,}B)"
        )

    contract_raw = manifest.get("contract") or {}
    try:
        contract = PipelineContract(
            input_shape=tuple(int(v) for v in contract_raw["input_shape"]),
            sample_interval_seconds=float(contract_raw["sample_interval_seconds"]),
            feature_fields=tuple(contract_raw.get("feature_fields", ())),
            normalization={
                name: (float(stats[0]), float(stats[1]))
                for name, stats in (contract_raw.get("normalization") or {}).items()
            },
            class_labels=tuple(contract_raw.get("class_labels", ())),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BundleRejected(
            f"계약이 불완전하다: {exc}. "
            "**전처리 명세 없이 모델만 받으면 조용히 틀린다** (실습 5-12, 5-15)"
        ) from exc

    if len(contract.class_labels) < 2:
        raise BundleRejected("라벨이 둘 미만이다. 분류 결과를 해석할 수 없다")

    rule_raw = manifest.get("alert_rule") or {}
    try:
        alert_rule = AlertRule(
            alert_labels=tuple(rule_raw.get("alert_labels", ())),
            dwell=int(rule_raw.get("dwell", 3)),
            min_confidence=float(rule_raw.get("min_confidence", 0.6)),
            cooldown_seconds=float(rule_raw.get("cooldown_seconds", 300.0)),
            hourly_budget=int(rule_raw.get("hourly_budget", 12)),
        )
    except Exception as exc:  # noqa: BLE001 - 경계에서 한 번에 번역한다
        raise BundleRejected(f"알람 규칙이 불완전하다: {exc}") from exc

    unknown = set(alert_rule.alert_labels) - set(contract.class_labels)
    if unknown:
        raise BundleRejected(
            f"알람 규칙이 모델에 없는 라벨을 가리킨다: {sorted(unknown)}. "
            "**이 알람은 영원히 안 뜬다.**"
        )

    return DeployedBundle(
        root=root,
        version=str(manifest.get("version", root.name)),
        model_version_id=str(manifest.get("model_version_id", "")),
        checksum=declared,
        contract=contract,
        alert_rule=alert_rule,
        expected_p95_ms=float(manifest.get("expected_p95_ms", 0.0)),
    )


def write_bundle(
    root: Path,
    *,
    version: str,
    model_bytes: bytes,
    contract: PipelineContract,
    alert_rule: AlertRule,
    model_version_id: str = "",
    expected_p95_ms: float = 0.0,
) -> Path:
    """묶음을 디스크에 놓는다. 서버가 만들고 OTA 가 내려보내는 모양 그대로다.

    실습과 테스트에서 쓴다. 현장에서는 [6-8](../../docs/curriculum/6-8.md) 의 OTA 가 이 자리를 채운다.
    """
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / MODEL
    model_path.write_bytes(model_bytes)
    (root / MANIFEST).write_text(
        json.dumps(
            {
                "version": version,
                "model_version_id": model_version_id,
                "checksum": checksum_of(model_path),
                "artifact_bytes": model_path.stat().st_size,
                "expected_p95_ms": expected_p95_ms,
                "contract": {
                    "input_shape": list(contract.input_shape),
                    "sample_interval_seconds": contract.sample_interval_seconds,
                    "feature_fields": list(contract.feature_fields),
                    "normalization": {
                        name: list(stats)
                        for name, stats in contract.normalization.items()
                    },
                    "class_labels": list(contract.class_labels),
                },
                "alert_rule": {
                    "alert_labels": list(alert_rule.alert_labels),
                    "dwell": alert_rule.dwell,
                    "min_confidence": alert_rule.min_confidence,
                    "cooldown_seconds": alert_rule.cooldown_seconds,
                    "hourly_budget": alert_rule.hourly_budget,
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return root
