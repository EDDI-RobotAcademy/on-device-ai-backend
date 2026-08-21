"""에이전트 테스트 준비.

에이전트는 백엔드가 아니므로 `edge-agent/` 를 import 경로에 넣어야 한다.
**그리고 그것만 넣는다** — 에이전트가 `infrastructure` 를 못 보게 하려는 것이 아니라,
보드에서의 배치를 그대로 흉내내기 위해서다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
AGENT = ROOT / "edge-agent"
if str(AGENT) not in sys.path:
    sys.path.insert(0, str(AGENT))


@pytest.fixture(scope="session")
def deployed_bundle(optimized, tmp_path_factory):  # noqa: ANN001, ANN201
    """모듈 4 가 고른 **진짜 TFLite 결과물**을 묶음으로 만든다.

    가짜 모델 파일을 쓰지 않는다. 그러면 계약 검증도 추론도 흉내가 된다.
    """
    from device_agent.bundle import write_bundle

    from domain.model.identifiers import TrainingRunId
    from domain.operations.alerting import AlertRule
    from domain.operations.pipeline import PipelineContract
    from domain.optimization.identifiers import OptimizationRunId
    from domain.optimization.optimization_run import OptimizationStatus
    from tests.support import optimization_scenario as os4

    run = optimized.optimization.runs.find_by_id(
        OptimizationRunId.of(optimized.run_id)
    )
    if run.status is not OptimizationStatus.SELECTED:
        os4.select(optimized.optimization, optimized.run_id)
        run = optimized.optimization.runs.find_by_id(
            OptimizationRunId.of(optimized.run_id)
        )

    selected = run.certificate.selected_artifact_id
    candidate = next(
        c for c in run.candidates if c.artifact.artifact_id == selected
    )
    training_run = optimized.optimization.training_runs.find_by_id(
        TrainingRunId.of(optimized.trained.run_id)
    )
    model = optimized.optimization.registry.get(training_run.model_version_id)

    contract = PipelineContract(
        input_shape=tuple(training_run.architecture.input_spec.shape),
        sample_interval_seconds=10.0,
        feature_fields=tuple(training_run.data.feature_fields),
        normalization=dict(training_run.data.normalization),
        class_labels=model.dataset.labels,
    )
    rule = AlertRule(
        alert_labels=("FAULT", "OVERLOAD"),
        dwell=3,
        min_confidence=0.6,
        cooldown_seconds=300.0,
        hourly_budget=12,
    )

    root = tmp_path_factory.mktemp("slots") / "a"
    write_bundle(
        root,
        version="v1.3.0",
        model_bytes=Path(candidate.artifact.uri).read_bytes(),
        contract=contract,
        alert_rule=rule,
        model_version_id=str(training_run.model_version_id),
        expected_p95_ms=candidate.benchmark.p95_ms,
    )
    return root
