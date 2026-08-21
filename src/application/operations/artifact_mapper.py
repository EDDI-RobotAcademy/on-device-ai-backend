"""Optimization → Operations 번역기 (Anti-Corruption Layer). (실습 5-1)

Operations Context 는 `OptimizationRun` 도 `TrainingRun` 도 모른다.

    OptimizationRun (모듈 4)  →  DeployedArtifactRef

번역하면서 세 곳에서 정보를 모은다.

    선택 판정   모듈 4 의 OptimizationCertificate
    기준 숫자   모듈 4 의 벤치마크 + 평가 (현장과 비교할 대상)
    전처리      모듈 3 의 TrainingDataRef — **결과물 파일에는 들어 있지 않다**

세 번째가 자주 빠진다. TFLite 파일에는 정규화 통계가 없다.
그것을 함께 보내지 않으면 디바이스가 다른 전처리를 하게 된다.
"""

from __future__ import annotations

from application.shared.errors import UnsupportedOperation
from domain.model.training_run import TrainingRun
from domain.operations.artifact_ref import DeployedArtifactRef
from domain.optimization.optimization_run import OptimizationRun


def artifact_from(
    run: OptimizationRun,
    training_run: TrainingRun | None = None,
    *,
    label: str | None = None,
) -> DeployedArtifactRef:
    """최적화 결과에서 배포 대상을 뽑아낸다.

    label 을 주지 않으면 **선택된 후보**를 쓴다.
    선택하지 않은 후보도 번역은 된다 — 막는 것은 `Deployment.deploy()` 의 일이다.
    """
    certificate = run.certificate
    target_label = label or (certificate.selected_label if certificate else None)
    if target_label is None:
        raise UnsupportedOperation(
            "선택된 결과물이 없다. 무엇을 배포할지 정하지 않았다.",
            subject=str(run.id),
        )

    candidate = next(
        (c for c in run.tradeoff_table().all_candidates if c.label == target_label),
        None,
    )
    if candidate is None:
        raise UnsupportedOperation(
            f"'{target_label}' 후보가 이 최적화에 없다.", subject=target_label
        )

    baseline = run.baseline
    selected = bool(
        certificate
        and certificate.has_selection
        and certificate.selected_label == target_label
    )

    return DeployedArtifactRef(
        artifact_id=str(candidate.artifact.artifact_id),
        optimization_run_ref=str(run.id),
        model_version_id=baseline.model_version_id,
        runtime=candidate.artifact.runtime.value,
        precision=candidate.artifact.precision.value,
        size_bytes=candidate.artifact.size_bytes,
        class_labels=baseline.class_labels,
        input_fields=tuple(training_run.data.feature_fields) if training_run else (),
        expected_p95_ms=candidate.benchmark.p95_ms,
        expected_accuracy=candidate.accuracy.accuracy,
        expected_class_mix=_expected_mix(candidate, baseline.class_labels),
        normalization=dict(training_run.data.normalization) if training_run else {},
        selected=selected,
    )


def _expected_mix(candidate, labels: tuple[str, ...]) -> dict[str, float]:  # noqa: ANN001
    """평가 때 이 모델이 각 클래스를 답한 비율.

    클래스별 재현율이 아니라 **예측 비율**이 필요하다.
    현장에서 셀 수 있는 것은 정답이 아니라 예측이기 때문이다.

    비어 있으면 비운 채로 넘긴다. 없는 기준을 지어내지 않는다 —
    `ReleasePolicy` 가 "기준이 없다"고 지적하게 두는 편이 낫다.
    """
    return {
        label: share
        for label, share in candidate.accuracy.predicted_mix.items()
        if label in labels
    }
