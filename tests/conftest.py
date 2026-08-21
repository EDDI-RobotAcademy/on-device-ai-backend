"""테스트 공통 준비.

실습 데이터는 세션당 한 번만 만들어 tmp 에 둔다.
data/samples/ 를 건드리지 않으므로, 학생이 그 파일을 편집하며 실습해도 테스트는 영향받지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from infrastructure.config.container import (
    DataContainer,
    DataQualityContainer,
    ModelContainer,
)
from infrastructure.monitoring.event_log import RecordingEventPublisher
from infrastructure.persistence.in_memory_assessment_repository import (
    InMemoryAssessmentRepository,
)
from infrastructure.persistence.in_memory_training_run_repository import (
    InMemoryTrainingRunRepository,
)
from infrastructure.persistence.in_memory_dataset_repository import (
    InMemoryDatasetRepository,
)
from infrastructure.sample_data import (
    CastingImageSample,
    IndustrialImageSample,
    ModelSample,
    PlantPowerSample,
    QualitySample,
    write_casting_images,
    write_industrial_images,
    write_model_samples,
    write_plant_power_samples,
    write_quality_samples,
)


@pytest.fixture(scope="session")
def sample_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("samples")


@pytest.fixture(scope="session")
def power(sample_root: Path) -> PlantPowerSample:
    """공장 전력 시계열 4종."""
    return write_plant_power_samples(sample_root)


@pytest.fixture(scope="session")
def castings(sample_root: Path) -> CastingImageSample:
    """다이캐스팅 부품 이미지."""
    return write_casting_images(sample_root)


@pytest.fixture
def events() -> RecordingEventPublisher:
    return RecordingEventPublisher()


@pytest.fixture
def container(events: RecordingEventPublisher) -> DataContainer:
    """실제 pandas/Pillow 어댑터를 쓰되, 저장소와 이벤트만 테스트용으로 바꾼다."""
    return DataContainer(
        repository=InMemoryDatasetRepository(),
        publisher=events,
    )


@pytest.fixture(scope="session")
def quality(sample_root: Path) -> QualitySample:
    """모듈 2용 전력 시계열 (구조는 멀쩡, 내용이 오염된 것과 그 기준선)."""
    return write_quality_samples(sample_root)


@pytest.fixture
def quality_container(
    container: DataContainer, events: RecordingEventPublisher
) -> DataQualityContainer:
    """Dataset 저장소를 Data Container 와 공유한다.

    품질 평가는 Dataset 을 번역해서 시작하므로, 두 Context 가 같은 저장소를 봐야 한다.
    """
    return DataQualityContainer(
        datasets=container.repository,
        assessments=InMemoryAssessmentRepository(),
        publisher=events,
    )


@pytest.fixture(scope="session")
def model_data(sample_root: Path) -> ModelSample:
    """모듈 3용 학습 데이터 + 현장 홀드아웃."""
    return write_model_samples(sample_root)


@pytest.fixture
def model_container(
    container: DataContainer,
    quality_container: DataQualityContainer,
    events: RecordingEventPublisher,
) -> ModelContainer:
    """Dataset/Assessment 저장소를 앞 모듈과 공유한다.

    학습을 준비하려면 두 게이트의 통과 여부를 읽어야 하기 때문이다.
    """
    return ModelContainer(
        datasets=container.repository,
        assessments=quality_container.assessments,
        runs=InMemoryTrainingRunRepository(),
        publisher=events,
    )


@pytest.fixture(scope="session")
def trained(model_data: ModelSample):  # noqa: ANN201
    """모듈 3 파이프라인을 세션당 한 번만 돌린다.

    학습은 느리다. 실습 3-1 ~ 3-9 는 이 결과를 **읽기만** 한다.
    상태를 바꾸는 실습(3-10)은 자기 파이프라인을 따로 만든다.
    """
    from tests.support.model_scenario import build_pipeline

    return build_pipeline(train_path=model_data.train)


@pytest.fixture(scope="session")
def trained_early_stop(model_data: ModelSample):  # noqa: ANN201
    """조기 종료를 건 학습. (실습 3-7)"""
    from domain.model.training_config import EarlyStoppingRule
    from tests.support.model_scenario import build_pipeline, training_config

    return build_pipeline(
        dataset_id="power-model-es",
        assessment_id="qa-power-model-es",
        run_id="run-early-stop",
        train_path=model_data.train,
        config=training_config(
            epochs=30, early_stopping=EarlyStoppingRule(patience=3)
        ),
    )


@pytest.fixture(scope="session")
def trained_disjoint(model_data: ModelSample):  # noqa: ANN201
    """창을 겹치지 않게 자른 학습. (실습 3-8, 3-10)"""
    from tests.support.model_scenario import build_pipeline

    return build_pipeline(
        dataset_id="power-model-disjoint",
        assessment_id="qa-power-model-disjoint",
        run_id="run-disjoint",
        train_path=model_data.train,
        stride=30,
    )


# ---------------------------------------------------------------------------
# 모듈 3 — 실험 비교 (실습 3-12, 3-14, 3-15)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def structure_lab(model_data: ModelSample):  # noqa: ANN201
    """창 길이만 바꿔 가며 네 번 학습한 결과. (실습 3-12)"""
    from tests.support.model_scenario import build_structure_lab

    return build_structure_lab(model_data.train)


@pytest.fixture(scope="session")
def hyperparameter_lab(model_data: ModelSample):  # noqa: ANN201
    """학습 설정만 바꿔 가며 네 번 학습한 결과. (실습 3-14)"""
    from tests.support.model_scenario import build_hyperparameter_lab

    return build_hyperparameter_lab(model_data.train)


@pytest.fixture(scope="session")
def quality_impact_lab(quality: QualitySample):  # noqa: ANN201
    """오염된 데이터와 정제된 데이터로 각각 학습한 결과. (실습 3-15)"""
    from tests.support.model_scenario import build_quality_impact_lab

    return build_quality_impact_lab(
        dirty_path=quality.dirty,
        clean_path=quality.clean,
        holdout_path=quality.holdout,
    )


# ---------------------------------------------------------------------------
# 모듈 3 — 이미지 (실습 3-11)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def industrial_images(sample_root: Path) -> IndustrialImageSample:
    """학습이 되는 크기의 산업 이미지. 두 캡스톤 주제에 각각 대응한다."""
    return write_industrial_images(sample_root)


@pytest.fixture(scope="session")
def trained_casting(industrial_images: IndustrialImageSample):  # noqa: ANN201
    """다이캐스팅 표면 균열 판별 모델. (캡스톤 주제 1)"""
    from tests.support.image_scenario import build_pipeline

    return build_pipeline(
        industrial_images.casting_root,
        class_count=2,
        run_id="run-casting",
        dataset_ref="castings",
    )


@pytest.fixture(scope="session")
def trained_food_average(industrial_images: IndustrialImageSample):  # noqa: ANN201
    """식품 표면 3분류. 전역 **평균** 풀링. (캡스톤 주제 2)"""
    from tests.support.image_scenario import build_pipeline

    return build_pipeline(
        industrial_images.food_root,
        class_count=3,
        run_id="run-food-avg",
        dataset_ref="food",
    )


@pytest.fixture(scope="session")
def trained_food_max(industrial_images: IndustrialImageSample):  # noqa: ANN201
    """같은 데이터, 같은 구조. 전역 **최대** 풀링만 다르다."""
    from domain.model.architecture import GlobalPooling
    from tests.support.image_scenario import build_pipeline

    return build_pipeline(
        industrial_images.food_root,
        class_count=3,
        run_id="run-food-max",
        dataset_ref="food",
        pooling=GlobalPooling.MAX,
    )


# ---------------------------------------------------------------------------
# 모듈 4 — 최적화
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def optimized(model_data: ModelSample, tmp_path_factory: pytest.TempPathFactory):  # noqa: ANN201
    """모듈 4 파이프라인을 세션당 한 번만 돌린다.

    학습 + 변환 5가지 + 벤치마크는 느리다.
    실습 4-1 ~ 4-9 는 이 결과를 **읽기만** 한다.
    상태를 바꾸는 실습(4-10)은 선택을 직접 호출한다 —
    선택은 이미 잰 숫자만 보므로 다시 돌려도 빠르다.
    """
    from tests.support.optimization_scenario import build_accepted_model, build_pipeline

    trained = build_accepted_model(model_data.train)
    artifacts = tmp_path_factory.mktemp("artifacts")
    return build_pipeline(trained, artifacts)


@pytest.fixture
def optimization_container(optimized, tmp_path: Path):  # noqa: ANN001, ANN201
    """상태를 바꿔도 되는 조립품. 모델 registry 는 세션 것을 공유한다.

    학습을 다시 하지 않기 위해서다. 저장소만 새로 만든다.
    """
    from infrastructure.config.container import OptimizationContainer
    from infrastructure.persistence.in_memory_optimization_run_repository import (
        InMemoryOptimizationRunRepository,
    )

    source = optimized.optimization
    return OptimizationContainer(
        training_runs=source.training_runs,
        registry=source.registry,
        runs=InMemoryOptimizationRunRepository(),
        publisher=source.publisher,
        artifact_dir=tmp_path / "artifacts",
    )


@pytest.fixture(scope="session")
def reduced(optimized):  # noqa: ANN001, ANN201
    """구조 축소 다섯 가지를 세션당 한 번만 적용한다. (실습 4-11)"""
    from tests.support.optimization_scenario import reduce_structure

    return reduce_structure(optimized.optimization, optimized.run_id)


# ---------------------------------------------------------------------------
# 모듈 5 — 운영
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def operations_data(sample_root: Path):  # noqa: ANN201
    """모듈 5용 4일치 현장 신호."""
    from infrastructure.sample_data import write_operations_samples

    return write_operations_samples(sample_root)


@pytest.fixture(scope="session")
def deployed(optimized, operations_data, tmp_path_factory: pytest.TempPathFactory):  # noqa: ANN001, ANN201
    """모듈 5 파이프라인을 세션당 한 번만 돌린다.

    배포 → 4일치 재생(추론 34,533건) → 창 12개 관측까지 한 번에.
    실습 5-1 ~ 5-7 은 이 결과를 **읽기만** 한다.
    상태를 바꾸는 실습(5-8, 5-10)은 자기 컨테이너를 따로 만든다.
    """
    from tests.support.operations_scenario import build_pipeline

    return build_pipeline(optimized, optimized.trained, operations_data.stream)


@pytest.fixture(scope="session")
def device_pipeline(optimized, operations_data):  # noqa: ANN001, ANN201
    """디바이스 안에서 도는 다섯 단계. (실습 5-12, 5-13)"""
    from tests.support.operations_scenario import build_device_pipeline

    return build_device_pipeline(
        optimized, optimized.trained, operations_data.stream
    )


@pytest.fixture
def operations_container(deployed):  # noqa: ANN001, ANN201
    """상태를 바꿔도 되는 조립품.

    로그(34,533건)와 측정기는 세션 것을 공유하고, 저장소만 새로 만든다.
    같은 ID 로 다시 배포해 두므로 로그가 그대로 붙는다.
    """
    from tests.support.operations_scenario import clone

    return clone(deployed)


# ---------------------------------------------------------------------------
# 모듈 6 — AWS
# ---------------------------------------------------------------------------
@pytest.fixture
def aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """가짜 자격증명.

    **진짜 자격증명이 환경에 있으면 실수로 진짜 AWS 를 부를 수 있다.**
    moto 를 켜기 전에 반드시 덮어쓴다.
    """
    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
    ):
        monkeypatch.setenv(name, "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-northeast-2")
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture
def aws(aws_credentials):  # noqa: ANN001, ANN201
    """moto 를 켠다. **이 안에서 boto3 호출은 메모리로 간다.**

    가짜 클라이언트가 아니라 **진짜 boto3 요청**을 가로챈다 —
    API 이름이 틀리면 여기서 터진다.
    """
    from moto import mock_aws

    with mock_aws():
        yield


@pytest.fixture
def fleet_env(aws):  # noqa: ANN001, ANN201
    """플릿 생성 → 업링크 → 데이터셋 → 학습 → 릴리스까지 끝난 상태.

    ML 학습이 없으므로 빠르다. 매 테스트마다 새로 만들어 서로 간섭하지 않는다.
    """
    from tests.support.fleet_scenario import build_pipeline

    return build_pipeline()


@pytest.fixture
def fleet_bare(aws):  # noqa: ANN001, ANN201
    """AWS 어댑터만 붙은 빈 컨테이너. 처음부터 만들어 보는 실습용."""
    from infrastructure.config.container import FleetContainer
    from infrastructure.edge.ota_simulator import SimulatedFleetOtaGateway
    from tests.support.fleet_scenario import aws_config, response_profile

    container = FleetContainer.with_aws(aws_config())
    container.ota = SimulatedFleetOtaGateway(container.ota, response_profile())
    return container
