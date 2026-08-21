"""아키텍처 규칙을 코드로 강제한다. (CLAUDE.md §2, §14, §15)

    interfaces → application → domain ← infrastructure

문서에만 적힌 규칙은 반드시 깨진다.
"PyTorch 는 Domain 이 아니다"를 사람이 기억하는 대신 테스트가 기억하게 한다.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

FORBIDDEN_IN_DOMAIN = {
    # 기술 프레임워크
    "pandas",
    "numpy",
    "scipy",
    "sklearn",
    "PIL",
    "torch",
    "torchvision",
    "onnx",
    "onnxruntime",
    "tensorflow",
    "boto3",
    "botocore",
    "sqlalchemy",
    "structlog",
    "fastapi",
    "starlette",
    "pydantic",
    # 바깥 레이어
    "application",
    "infrastructure",
    "interfaces",
}

FORBIDDEN_IN_APPLICATION = {
    "pandas",
    "numpy",
    "scipy",
    "sklearn",
    "PIL",
    "torch",
    "onnx",
    "tensorflow",
    "boto3",
    "sqlalchemy",
    "fastapi",
    "starlette",
    "pydantic",
    "infrastructure",
    "interfaces",
}

FORBIDDEN_IN_INFRASTRUCTURE = {"fastapi", "starlette", "interfaces"}


def _modules(package: str) -> list[Path]:
    return sorted((SRC / package).rglob("*.py"))


def _imported_roots(path: Path) -> set[str]:
    """이 파일이 import 하는 최상위 패키지 이름들."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def _check(package: str, forbidden: set[str]) -> None:
    violations: list[str] = []
    for path in _modules(package):
        offending = _imported_roots(path) & forbidden
        if offending:
            relative = path.relative_to(SRC)
            violations.append(f"{relative}: {sorted(offending)}")
    assert not violations, "레이어 규칙 위반:\n  " + "\n  ".join(violations)


def test_domain_은_어떤_기술도_모른다() -> None:
    """PyTorch 도, pandas 도, FastAPI 도, AWS 도 Domain 이 아니다."""
    _check("domain", FORBIDDEN_IN_DOMAIN)


def _cross_context_imports(package: str, forbidden_prefixes: tuple[str, ...]) -> list[str]:
    violations: list[str] = []
    for path in _modules(package):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
            elif isinstance(node, ast.Import):
                module = node.names[0].name
            if module and module.startswith(forbidden_prefixes):
                violations.append(f"{path.relative_to(SRC)}: {module}")
    return violations


@pytest.mark.parametrize(
    ("package", "forbidden"),
    [
        ("domain/data_quality", ("domain.data.",)),
        ("domain/model", ("domain.data.", "domain.data_quality.")),
        (
            "domain/optimization",
            ("domain.data.", "domain.data_quality.", "domain.model."),
        ),
        (
            "domain/operations",
            (
                "domain.data.",
                "domain.data_quality.",
                "domain.model.",
                "domain.optimization.",
            ),
        ),
        (
            "domain/fleet",
            (
                "domain.data.",
                "domain.data_quality.",
                "domain.model.",
                "domain.optimization.",
                "domain.operations.",
            ),
        ),
    ],
)
def test_Bounded_Context_는_서로를_모른다(package: str, forbidden: tuple[str, ...]) -> None:
    """각 Context 는 다른 Context 의 Domain 을 import 하지 않는다.

    필요한 정보는 원시 값만 담은 VO 로 번역되어 들어온다.
        data_quality ← AssessmentTarget
        model        ← TrainingDataRef
        optimization ← BaselineModelRef
        operations   ← DeployedArtifactRef
        fleet        ← ReleaseBundle
    번역은 Application Layer(*_mapper.py)의 책임이다.
    """
    violations = _cross_context_imports(package, forbidden)
    assert not violations, (
        f"{package} 가 다른 Context 를 직접 참조한다:\n  " + "\n  ".join(violations)
    )


ML_FRAMEWORK_HOMES: dict[str, tuple[tuple[str, str], ...]] = {
    # 학습은 infrastructure/ml 이, 변환은 infrastructure/optimization 이 맡는다.
    # 변환의 출발점이 학습된 torch 모델이므로 두 곳 모두 torch 를 안다.
    "torch": (("infrastructure", "ml"), ("infrastructure", "optimization")),
    "torchvision": (("infrastructure", "ml"),),
    # 변환 대상 형식들. 오직 변환 어댑터만 안다.
    "onnx": (("infrastructure", "optimization"),),
    "onnxruntime": (("infrastructure", "optimization"),),
    "tensorflow": (("infrastructure", "optimization"),),
    "ai_edge_litert": (("infrastructure", "optimization"),),
    # CLAUDE.md §15 — AWS 는 Infrastructure 다. **이 규칙이 모듈 6 의 핵심이다.**
    "boto3": (("infrastructure", "aws"),),
    "botocore": (("infrastructure", "aws"),),
    "moto": ((),),
}


@pytest.mark.parametrize("framework", sorted(ML_FRAMEWORK_HOMES))
def test_ML_프레임워크는_정해진_디렉터리_밖으로_나가지_않는다(framework: str) -> None:
    """CLAUDE.md §14 — PyTorch 도 ONNX 도 TFLite 도 Domain 이 아니다.

    프레임워크를 바꾸면 고치는 곳이 한 디렉터리로 끝나야 한다.
    """
    homes = ML_FRAMEWORK_HOMES[framework]
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC)
        if relative.parts[:2] in homes:
            continue
        if framework in _imported_roots(path):
            offenders.append(str(relative))
    allowed = " 또는 ".join("/".join(home) for home in homes)
    assert not offenders, (
        f"{framework} 가 {allowed} 밖에서 쓰인다:\n  " + "\n  ".join(offenders)
    )


def test_application_은_구현체를_모른다() -> None:
    """Use Case 는 Port 만 본다. 어떤 어댑터가 끼워졌는지 알지 못한다."""
    _check("application", FORBIDDEN_IN_APPLICATION)


def test_infrastructure_는_HTTP_를_모른다() -> None:
    """어댑터가 HTTP 응답을 만들기 시작하면 재사용이 불가능해진다."""
    _check("infrastructure", FORBIDDEN_IN_INFRASTRUCTURE)


def test_domain_의_모든_모듈은_import_가능하다() -> None:
    """순환 의존이 있으면 여기서 터진다."""
    import importlib

    for path in _modules("domain"):
        module = ".".join(path.relative_to(SRC).with_suffix("").parts)
        if module.endswith(".__init__"):
            module = module.removesuffix(".__init__")
        importlib.import_module(module)


@pytest.mark.parametrize(
    "port_name",
    [
        "DatasetRepository",
        "DatasetProfiler",
        "SchemaInferrer",
        "SensorSignalMeasurer",
        "ImageSignalMeasurer",
        "TimeAxisMeasurer",
        "LabelMeasurer",
        "PartitionEngine",
        "NormalizationFitter",
        "DistributionComparer",
    ],
)
def test_모든_Port_에_실제_구현이_조립되어_있다(port_name: str) -> None:
    """Port 만 있고 구현이 없는 상태로 방치되지 않게 한다."""
    from infrastructure.config.container import DataContainer

    container = DataContainer()
    attribute = {
        "DatasetRepository": "repository",
        "DatasetProfiler": "profiler",
        "SchemaInferrer": "schema_inferrer",
        "SensorSignalMeasurer": "sensor_measurer",
        "ImageSignalMeasurer": "image_measurer",
        "TimeAxisMeasurer": "time_axis_measurer",
        "LabelMeasurer": "label_measurer",
        "PartitionEngine": "partition_engine",
        "NormalizationFitter": "normalization_fitter",
        "DistributionComparer": "distribution_comparer",
    }[port_name]
    assert getattr(container, attribute) is not None


@pytest.mark.parametrize(
    "attribute",
    [
        "assessments",
        "missing_measurer",
        "outlier_measurer",
        "label_measurer",
        "balance_measurer",
        "noise_measurer",
        "duplicate_measurer",
    ],
)
def test_품질_Port_에도_구현이_조립되어_있다(attribute: str) -> None:
    from infrastructure.config.container import DataQualityContainer

    assert getattr(DataQualityContainer(), attribute) is not None


def test_품질_측정기는_전부_Port_를_만족한다() -> None:
    """runtime_checkable Protocol 로 구조적 일치를 확인한다."""
    from domain.data_quality import ports
    from infrastructure.config.container import DataQualityContainer

    container = DataQualityContainer()
    pairs = (
        (container.assessments, ports.QualityAssessmentRepository),
        (container.missing_measurer, ports.MissingValueMeasurer),
        (container.outlier_measurer, ports.OutlierMeasurer),
        (container.label_measurer, ports.LabelErrorMeasurer),
        (container.balance_measurer, ports.ClassBalanceMeasurer),
        (container.noise_measurer, ports.NoiseMeasurer),
        (container.duplicate_measurer, ports.DuplicateMeasurer),
    )
    for adapter, protocol in pairs:
        assert isinstance(adapter, protocol), f"{type(adapter).__name__} ≠ {protocol.__name__}"


@pytest.mark.parametrize(
    "attribute",
    ["runs", "materializer", "image_materializer", "profiler", "trainer", "evaluator"],
)
def test_모델_Port_에도_구현이_조립되어_있다(attribute: str) -> None:
    from infrastructure.config.container import ModelContainer

    assert getattr(ModelContainer(), attribute) is not None


def test_모델_어댑터는_전부_Port_를_만족한다() -> None:
    from domain.model import ports
    from infrastructure.config.container import ModelContainer

    container = ModelContainer()
    pairs = (
        (container.runs, ports.TrainingRunRepository),
        (container.materializer, ports.TensorMaterializer),
        (container.image_materializer, ports.ImageTensorMaterializer),
        (container.profiler, ports.ArchitectureProfiler),
        (container.trainer, ports.ModelTrainer),
        (container.evaluator, ports.ModelEvaluator),
        (container.evaluator, ports.FieldEvaluator),
    )
    for adapter, protocol in pairs:
        assert isinstance(adapter, protocol), (
            f"{type(adapter).__name__} ≠ {protocol.__name__}"
        )


@pytest.mark.parametrize(
    "attribute",
    ["runs", "exporter", "checker", "benchmarker", "accuracy", "roofline"],
)
def test_최적화_Port_에도_구현이_조립되어_있다(attribute: str) -> None:
    from infrastructure.config.container import OptimizationContainer

    assert getattr(OptimizationContainer(), attribute) is not None


def test_최적화_어댑터는_전부_Port_를_만족한다() -> None:
    from domain.optimization import ports
    from infrastructure.config.container import OptimizationContainer

    container = OptimizationContainer()
    pairs = (
        (container.runs, ports.OptimizationRunRepository),
        (container.exporter, ports.ModelExporter),
        (container.checker, ports.EquivalenceChecker),
        (container.benchmarker, ports.RuntimeBenchmarker),
        (container.accuracy, ports.ArtifactAccuracyEvaluator),
        (container.roofline, ports.RooflineProfiler),
    )
    for adapter, protocol in pairs:
        assert isinstance(adapter, protocol), (
            f"{type(adapter).__name__} ≠ {protocol.__name__}"
        )


@pytest.mark.parametrize(
    "attribute",
    ["deployments", "watches", "logs", "latency", "mix"],
)
def test_운영_Port_에도_구현이_조립되어_있다(attribute: str) -> None:
    from infrastructure.config.container import OperationsContainer

    assert getattr(OperationsContainer(), attribute) is not None


def test_운영_어댑터는_전부_Port_를_만족한다() -> None:
    from domain.operations import ports
    from infrastructure.config.container import OperationsContainer

    container = OperationsContainer()
    pairs = (
        (container.deployments, ports.DeploymentRepository),
        (container.watches, ports.HealthWatchRepository),
        (container.logs, ports.InferenceLogStore),
        (container.latency, ports.LatencyMeasurer),
        (container.mix, ports.PredictionMixMeasurer),
    )
    for adapter, protocol in pairs:
        assert isinstance(adapter, protocol), (
            f"{type(adapter).__name__} ≠ {protocol.__name__}"
        )


def test_드리프트_측정기가_없어도_조용히_통과하지_않는다() -> None:
    """학습 분포를 안 넘겨주면 빈 보고서가 나오고, Policy 가 그것을 지적한다."""
    from domain.operations.drift import DriftPolicy
    from domain.operations.window import ObservationWindow
    from infrastructure.config.container import OperationsContainer

    container = OperationsContainer()
    assert container.drift is None

    measurer = container.observe_health()._drift  # noqa: SLF001
    window = ObservationWindow(
        label="w",
        started_at="2026-05-20 00:00:00",
        ended_at="2026-05-20 07:59:59",
        sample_count=100,
    )
    report = measurer.measure(None, window)
    findings = DriftPolicy().inspect(report)
    assert "OPS_NO_DRIFT_BASELINE" in {f.code for f in findings}


# ---------------------------------------------------------------------------
# CLAUDE.md §15 — AWS Isolation
# ---------------------------------------------------------------------------
AWS_WORDS = (
    "boto3",
    "botocore",
    "s3",
    "S3",
    "dynamodb",
    "DynamoDB",
    "sagemaker",
    "SageMaker",
    "aws",
    "AWS",
    "IoT Jobs",
)


def _code_identifiers(path: Path) -> set[str]:
    """docstring 을 뺀 **코드에 실제로 등장하는 이름들.**

    docstring 은 오히려 AWS 를 이야기해야 한다 — "왜 여기 없는가" 를 설명하려면.
    검사해야 하는 것은 **코드가 AWS 를 아는가**이지 문서가 언급하는가가 아니다.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def test_domain_fleet_의_코드는_AWS_를_모른다() -> None:
    """**모듈 6 이 증명해야 하는 것.** (CLAUDE.md §15)

    docstring 은 AWS 를 이야기한다 — "S3 라고 부르지 않는다" 를 설명해야 하기 때문이다.
    그러나 **코드에는 그 어휘가 하나도 없다.**
    """
    forbidden = {
        "boto3",
        "botocore",
        "s3",
        "S3",
        "dynamodb",
        "DynamoDB",
        "sagemaker",
        "SageMaker",
        "iot",
        "IoT",
    }
    offenders: list[str] = []
    for path in sorted((SRC / "domain" / "fleet").rglob("*.py")):
        hit = _code_identifiers(path) & forbidden
        if hit:
            offenders.append(f"{path.relative_to(SRC)}: {sorted(hit)}")
    assert not offenders, (
        "domain/fleet 의 코드가 AWS 를 안다:\n  " + "\n  ".join(offenders)
    )


def test_docstring_은_오히려_AWS_를_설명한다() -> None:
    """규칙을 코드로 막는 것만으로는 부족하다. **왜 그런지가 적혀 있어야 한다.**"""
    ports_doc = (SRC / "domain" / "fleet" / "ports.py").read_text(encoding="utf-8")
    assert "boto3" in ports_doc
    assert "S3Client" in ports_doc


def test_Port_이름에도_AWS_가_없다() -> None:
    from domain.fleet import ports

    names = [name for name in dir(ports) if not name.startswith("_")]
    aws_named = [
        name
        for name in names
        if any(word.lower() in name.lower() for word in ("s3", "dynamo", "sagemaker", "iot"))
    ]
    assert not aws_named, f"Port 이름에 AWS 가 들어 있다: {aws_named}"


def test_어댑터가_Port_이름을_구현한다() -> None:
    """Port 는 `ObjectStore`, 어댑터는 `S3ObjectStore`.

    이름이 그대로 대응하면 어느 구멍에 무엇이 꽂혀 있는지 한눈에 보인다.
    """
    from infrastructure.aws.dynamo_device_registry import DynamoDeviceRegistry
    from infrastructure.aws.iot_ota_gateway import IotJobsOtaGateway
    from infrastructure.aws.s3_object_store import S3ObjectStore
    from infrastructure.aws.sagemaker_training import SageMakerTrainingGateway

    for adapter, port in (
        (S3ObjectStore, "ObjectStore"),
        (DynamoDeviceRegistry, "DeviceRegistry"),
        (SageMakerTrainingGateway, "TrainingGateway"),
        (IotJobsOtaGateway, "OtaGateway"),
    ):
        assert adapter.__name__.endswith(port), adapter.__name__


@pytest.mark.parametrize(
    "attribute", ["fleets", "rollouts", "clock"]
)
def test_플릿_Port_에도_구현이_조립되어_있다(attribute: str) -> None:
    from infrastructure.config.container import FleetContainer

    assert getattr(FleetContainer(), attribute) is not None


def test_AWS_없이도_컨테이너를_만들_수_있다() -> None:
    """**자격증명이 없는 환경에서 컨테이너를 만드는 것만으로 boto3 가 깨면 안 된다.**"""
    from application.shared.errors import UnsupportedOperation
    from infrastructure.config.container import FleetContainer

    container = FleetContainer()
    assert container.store is None
    with pytest.raises(UnsupportedOperation):
        container.ingest_uplink()


def test_플릿_어댑터는_전부_Port_를_만족한다(aws) -> None:  # noqa: ANN001
    from domain.fleet import ports
    from infrastructure.config.container import FleetContainer
    from tests.support.fleet_scenario import aws_config

    container = FleetContainer.with_aws(aws_config())
    pairs = (
        (container.fleets, ports.FleetRepository),
        (container.rollouts, ports.RolloutRepository),
        (container.store, ports.ObjectStore),
        (container.registry, ports.DeviceRegistry),
        (container.training, ports.RemoteTrainingGateway),
        (container.ota, ports.OtaGateway),
    )
    for adapter, protocol in pairs:
        assert isinstance(adapter, protocol), (
            f"{type(adapter).__name__} ≠ {protocol.__name__}"
        )


def test_시뮬레이터도_같은_Port_를_구현한다(aws) -> None:  # noqa: ANN001
    """**Domain 은 진짜와 흉내를 구분하지 못한다.** 그게 Port 를 둔 이유다."""
    from domain.fleet import ports
    from infrastructure.edge.ota_simulator import SimulatedFleetOtaGateway
    from infrastructure.config.container import FleetContainer
    from tests.support.fleet_scenario import aws_config

    container = FleetContainer.with_aws(aws_config())
    simulated = SimulatedFleetOtaGateway(container.ota)
    assert isinstance(simulated, ports.OtaGateway)


# ---------------------------------------------------------------------------
# 디바이스 에이전트 (edge-agent/) — 보드에 올라가는 것
# ---------------------------------------------------------------------------
EDGE_AGENT = Path(__file__).resolve().parents[1] / "edge-agent" / "device_agent"

FORBIDDEN_IN_EDGE_AGENT = {
    # 바깥 레이어 — 하나라도 끌어오면 무거운 것이 전부 따라온다
    "application",
    "infrastructure",
    "interfaces",
    # 보드에 올릴 수 없는 것들
    "pandas",
    "torch",
    "torchvision",
    "tensorflow",
    "sklearn",
    "scipy",
    "onnx",
    "onnxruntime",
    "boto3",
    "botocore",
    "fastapi",
    "starlette",
    "pydantic",
    "structlog",
    "moto",
}


def test_디바이스_에이전트는_domain_만_가져간다() -> None:
    """**이것이 이 구조의 값어치다.**

    `domain` 에는 프레임워크가 하나도 없다 (CLAUDE.md §14, §15).
    그래서 판정 코드를 **그대로 보드에 올릴 수 있다.**

        서버   AlertGate          사고를 재현할 때 (실습 5-13)
        보드   StreamingAlertGate 지금 판단할 때
        → 같은 규칙, 같은 결과

    `infrastructure` 를 하나라도 import 하면 pandas·torch·tensorflow 가
    따라 올라온다. 보드에는 그것을 올릴 자리가 없다.
    """
    violations: list[str] = []
    for path in sorted(EDGE_AGENT.rglob("*.py")):
        offending = _imported_roots(path) & FORBIDDEN_IN_EDGE_AGENT
        if offending:
            violations.append(f"{path.name} → {sorted(offending)}")

    assert not violations, "에이전트가 보드에 못 올릴 것을 가져간다:\n" + "\n".join(
        violations
    )


def test_디바이스_에이전트가_실제로_domain_을_쓴다() -> None:
    """안 쓰는 것으로 규칙을 지키는 것은 규칙을 지킨 것이 아니다.

    에이전트는 백엔드와 **같은 Domain 객체**로 판단해야 한다.
    별도의 알람 규칙을 하나 더 만들면 그 순간 두 곳의 판정이 갈린다.
    """
    used: set[str] = set()
    for path in sorted(EDGE_AGENT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "domain."
            ):
                used.update(alias.name for alias in node.names)

    for required in ("AlertRule", "StreamingAlertGate", "PipelineContract", "PipelineRun"):
        assert required in used, f"에이전트가 {required} 를 쓰지 않는다"


def test_에이전트의_의존성_목록에_무거운_것이_없다() -> None:
    """`edge-agent/requirements.txt` 가 보드에 올라갈 것의 전부다."""
    requirements = (
        Path(__file__).resolve().parents[1] / "edge-agent" / "requirements.txt"
    ).read_text(encoding="utf-8")
    declared = {
        line.split(">=")[0].split("==")[0].strip().lower()
        for line in requirements.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    heavy = declared & {"pandas", "torch", "tensorflow", "boto3", "fastapi", "scipy"}
    assert not heavy, f"보드에 올릴 수 없는 것이 목록에 있다: {sorted(heavy)}"
    assert "numpy" in declared
