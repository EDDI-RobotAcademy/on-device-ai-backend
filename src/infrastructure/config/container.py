"""의존성 조립.

어떤 구현체를 쓸지 결정하는 곳은 여기 한 군데뿐이다. (CLAUDE.md §20)
Application 도 Domain 도 "누가 실제로 측정하는지" 모른다.

교육 중 pandas 대신 가짜 측정기를 끼우고 싶으면 이 클래스만 갈아끼우면 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from application.data.analyze_representativeness import AnalyzeRepresentativeness
from application.data.certify_dataset_readiness import (
    CertifyDatasetReadiness,
    ReopenDataset,
)
from application.data.declare_data_schema import DeclareDataSchema
from application.data.define_label_space import DefineLabelSpace
from application.data.design_sampling import DesignSampling
from application.data.design_training_data import DesignTrainingData
from application.data.get_dataset import GetDataset, ListDatasets
from application.data.infer_data_schema import InferDataSchema
from application.data.inspect_signal_plausibility import InspectSignalPlausibility
from application.data.inspect_time_axis import InspectTimeAxis
from application.data.partition_dataset import PartitionDataset
from application.data.profile_dataset import ProfileDataset
from application.data.register_dataset import RegisterDataset
from application.shared.ports import EventPublisher
from domain.data.ports import (
    DatasetProfiler,
    DatasetRepository,
    DistributionComparer,
    ImageSignalMeasurer,
    LabelMeasurer,
    NormalizationFitter,
    PartitionEngine,
    SamplingProbe,
    SchemaInferrer,
    SensorSignalMeasurer,
    TimeAxisMeasurer,
)
from infrastructure.analysis.numpy_distribution_comparer import NumpyDistributionComparer
from infrastructure.analysis.pandas_dataset_profiler import (
    HeuristicSchemaInferrer,
    PandasDatasetProfiler,
)
from infrastructure.analysis.pandas_label_measurer import PandasLabelMeasurer
from infrastructure.analysis.pandas_normalization_fitter import (
    PandasNormalizationFitter,
)
from infrastructure.analysis.pandas_partition_engine import PandasPartitionEngine
from infrastructure.analysis.pandas_sampling_probe import PandasSamplingProbe
from infrastructure.analysis.pandas_sensor_signal_measurer import (
    PandasSensorSignalMeasurer,
)
from infrastructure.analysis.pandas_time_axis_measurer import PandasTimeAxisMeasurer
from infrastructure.analysis.pillow_image_signal_measurer import (
    PillowImageSignalMeasurer,
)
from infrastructure.monitoring.event_log import StructlogEventPublisher
from infrastructure.persistence.in_memory_assessment_repository import (
    InMemoryAssessmentRepository,
)
from infrastructure.persistence.in_memory_dataset_repository import (
    InMemoryDatasetRepository,
)
from application.model.accept_model import AcceptModel, ReopenTrainingRun
from application.model.evaluate_model import EvaluateModel, EvaluateModelOnField
from application.model.compare_experiments import CompareExperiments
from application.model.compare_with_baseline import CompareWithBaseline
from application.model.execute_image_training_run import ExecuteImageTrainingRun
from application.model.execute_training_run import ExecuteTrainingRun
from application.model.get_training_run import (
    GetTrainingCurve,
    GetTrainingRun,
    ListTrainingRuns,
)
from application.model.prepare_image_training_run import PrepareImageTrainingRun
from application.model.prepare_training_run import PrepareTrainingRun
from domain.model.ports import (
    ArchitectureProfiler,
    BaselineDetector,
    ImageFolderInspector,
    ImageModelTrainer,
    ImageTensorMaterializer,
    ModelEvaluator,
    ModelTrainer,
    TensorMaterializer,
    TrainingRunRepository,
)
from infrastructure.ml.image_dataset import PillowImageFolderInspector
from infrastructure.ml.statistical_detector import StatisticalAnomalyDetector
from infrastructure.ml.torch_architecture import TorchArchitectureProfiler
from infrastructure.ml.torch_image_trainer import PyTorchImageTrainer
from infrastructure.ml.torch_materializer import (
    PillowImageTensorMaterializer,
    TorchTensorMaterializer,
)
from infrastructure.ml.torch_trainer import (
    PyTorchModelTrainer,
    TorchModelEvaluator,
    TorchModelRegistry,
)
from infrastructure.persistence.in_memory_training_run_repository import (
    InMemoryTrainingRunRepository,
)

from application.data_quality.compare_quality import CompareQuality
from application.data_quality.compare_rebalancing import CompareRebalancing
from application.data_quality.evaluate_quality_gate import (
    EvaluateQualityGate,
    ReopenAssessment,
)
from application.data_quality.get_assessment import (
    GetAssessment,
    ListAssessments,
)
from application.data_quality.measure_balance import MeasureBalance
from application.data_quality.measure_completeness import (
    MeasureCompleteness,
)
from application.data_quality.measure_label_quality import (
    MeasureLabelQuality,
)
from application.data_quality.measure_noise import MeasureNoise
from application.data_quality.measure_uniqueness import (
    MeasureUniqueness,
)
from application.data_quality.measure_validity import MeasureValidity
from application.data_quality.record_remediation import (
    RecordRemediation,
)
from application.data_quality.score_quality import ScoreQuality
from application.data_quality.start_quality_assessment import (
    StartQualityAssessment,
)
from domain.data_quality.ports import (
    ClassBalanceMeasurer,
    DuplicateMeasurer,
    LabelErrorMeasurer,
    MissingValueMeasurer,
    NoiseMeasurer,
    OutlierMeasurer,
    QualityAssessmentRepository,
    Resampler,
)
from infrastructure.analysis.pandas_class_balance_measurer import (
    PandasClassBalanceMeasurer,
)
from infrastructure.analysis.pandas_duplicate_measurer import (
    PandasDuplicateMeasurer,
)
from infrastructure.analysis.pandas_label_error_measurer import (
    PandasLabelErrorMeasurer,
)
from infrastructure.analysis.pandas_missing_value_measurer import (
    PandasMissingValueMeasurer,
)
from infrastructure.analysis.pandas_noise_measurer import (
    PandasNoiseMeasurer,
)
from infrastructure.analysis.pandas_outlier_measurer import (
    PandasOutlierMeasurer,
)
from infrastructure.analysis.pandas_resampler import PandasResampler


@dataclass(slots=True)
class DataContainer:
    """Data Context 의 조립품."""

    repository: DatasetRepository = field(default_factory=InMemoryDatasetRepository)
    publisher: EventPublisher | None = field(default_factory=StructlogEventPublisher)
    profiler: DatasetProfiler = field(default_factory=PandasDatasetProfiler)
    schema_inferrer: SchemaInferrer = field(default_factory=HeuristicSchemaInferrer)
    sensor_measurer: SensorSignalMeasurer = field(
        default_factory=PandasSensorSignalMeasurer
    )
    image_measurer: ImageSignalMeasurer = field(
        default_factory=PillowImageSignalMeasurer
    )
    time_axis_measurer: TimeAxisMeasurer = field(
        default_factory=PandasTimeAxisMeasurer
    )
    label_measurer: LabelMeasurer = field(default_factory=PandasLabelMeasurer)
    partition_engine: PartitionEngine = field(default_factory=PandasPartitionEngine)
    normalization_fitter: NormalizationFitter = field(
        default_factory=PandasNormalizationFitter
    )
    distribution_comparer: DistributionComparer = field(
        default_factory=NumpyDistributionComparer
    )
    sampling_probe: SamplingProbe = field(default_factory=PandasSamplingProbe)

    # -- Use Case ---------------------------------------------------------
    def design_sampling(self) -> DesignSampling:
        return DesignSampling(self.repository, self.sampling_probe)

    def register_dataset(self) -> RegisterDataset:
        return RegisterDataset(self.repository, self.publisher)

    def profile_dataset(self) -> ProfileDataset:
        return ProfileDataset(self.repository, self.profiler, self.publisher)

    def infer_data_schema(self) -> InferDataSchema:
        return InferDataSchema(self.repository, self.schema_inferrer)

    def declare_data_schema(self) -> DeclareDataSchema:
        return DeclareDataSchema(self.repository, self.publisher)

    def inspect_signal_plausibility(self) -> InspectSignalPlausibility:
        return InspectSignalPlausibility(
            self.repository, self.sensor_measurer, self.image_measurer, self.publisher
        )

    def inspect_time_axis(self) -> InspectTimeAxis:
        return InspectTimeAxis(self.repository, self.time_axis_measurer, self.publisher)

    def define_label_space(self) -> DefineLabelSpace:
        return DefineLabelSpace(self.repository, self.label_measurer, self.publisher)

    def design_training_data(self) -> DesignTrainingData:
        return DesignTrainingData(
            self.repository, self.normalization_fitter, self.publisher
        )

    def partition_dataset(self) -> PartitionDataset:
        return PartitionDataset(self.repository, self.partition_engine, self.publisher)

    def analyze_representativeness(self) -> AnalyzeRepresentativeness:
        return AnalyzeRepresentativeness(
            self.repository, self.distribution_comparer, self.publisher
        )

    def certify_dataset_readiness(self) -> CertifyDatasetReadiness:
        return CertifyDatasetReadiness(self.repository, self.publisher)

    def reopen_dataset(self) -> ReopenDataset:
        return ReopenDataset(self.repository, self.publisher)

    def get_dataset(self) -> GetDataset:
        return GetDataset(self.repository)

    def list_datasets(self) -> ListDatasets:
        return ListDatasets(self.repository)


# ---------------------------------------------------------------------------
# 모듈 2 — Data Quality Context
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class DataQualityContainer:
    """Data Quality Context 의 조립품.

    Dataset 저장소를 함께 받는다 — 품질 평가를 시작하려면 Dataset 을
    AssessmentTarget 으로 번역해야 하고, 그 번역은 Application 의 일이기 때문이다.
    """

    datasets: DatasetRepository = field(default_factory=InMemoryDatasetRepository)
    assessments: QualityAssessmentRepository = field(
        default_factory=InMemoryAssessmentRepository
    )
    publisher: EventPublisher | None = field(default_factory=StructlogEventPublisher)

    missing_measurer: MissingValueMeasurer = field(
        default_factory=PandasMissingValueMeasurer
    )
    outlier_measurer: OutlierMeasurer = field(default_factory=PandasOutlierMeasurer)
    label_measurer: LabelErrorMeasurer = field(
        default_factory=PandasLabelErrorMeasurer
    )
    balance_measurer: ClassBalanceMeasurer = field(
        default_factory=PandasClassBalanceMeasurer
    )
    noise_measurer: NoiseMeasurer = field(default_factory=PandasNoiseMeasurer)
    duplicate_measurer: DuplicateMeasurer = field(
        default_factory=PandasDuplicateMeasurer
    )
    resampler: Resampler = field(default_factory=PandasResampler)

    # -- Use Case ---------------------------------------------------------
    def compare_rebalancing(self) -> CompareRebalancing:
        return CompareRebalancing(self.datasets, self.resampler)

    def start_quality_assessment(self) -> StartQualityAssessment:
        return StartQualityAssessment(self.assessments, self.datasets, self.publisher)

    def measure_completeness(self) -> MeasureCompleteness:
        return MeasureCompleteness(
            self.assessments, self.missing_measurer, self.publisher
        )

    def measure_validity(self) -> MeasureValidity:
        return MeasureValidity(self.assessments, self.outlier_measurer, self.publisher)

    def measure_label_quality(self) -> MeasureLabelQuality:
        return MeasureLabelQuality(
            self.assessments, self.label_measurer, self.publisher
        )

    def measure_balance(self) -> MeasureBalance:
        return MeasureBalance(self.assessments, self.balance_measurer, self.publisher)

    def measure_noise(self) -> MeasureNoise:
        return MeasureNoise(self.assessments, self.noise_measurer, self.publisher)

    def measure_uniqueness(self) -> MeasureUniqueness:
        return MeasureUniqueness(
            self.assessments, self.duplicate_measurer, self.publisher
        )

    def score_quality(self) -> ScoreQuality:
        return ScoreQuality(
            self.assessments,
            self.missing_measurer,
            self.duplicate_measurer,
            self.balance_measurer,
            self.label_measurer,
            self.publisher,
        )

    def compare_quality(self) -> CompareQuality:
        return CompareQuality(self.assessments, self.publisher)

    def record_remediation(self) -> RecordRemediation:
        return RecordRemediation(self.assessments, self.publisher)

    def evaluate_quality_gate(self) -> EvaluateQualityGate:
        return EvaluateQualityGate(self.assessments, self.publisher)

    def reopen_assessment(self) -> ReopenAssessment:
        return ReopenAssessment(self.assessments, self.publisher)

    def get_assessment(self) -> GetAssessment:
        return GetAssessment(self.assessments)

    def list_assessments(self) -> ListAssessments:
        return ListAssessments(self.assessments)


# ---------------------------------------------------------------------------
# 모듈 3 — Model Context
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ModelContainer:
    """Model Context 의 조립품.

    Dataset / QualityAssessment 저장소를 함께 받는다 —
    학습을 준비하려면 두 게이트의 통과 여부를 읽어야 하기 때문이다.

    PyTorch 라는 단어가 이 프로젝트에서 등장하는 곳은 여기와 infrastructure/ml 뿐이다.
    """

    datasets: DatasetRepository = field(default_factory=InMemoryDatasetRepository)
    assessments: QualityAssessmentRepository = field(
        default_factory=InMemoryAssessmentRepository
    )
    runs: TrainingRunRepository = field(
        default_factory=InMemoryTrainingRunRepository
    )
    publisher: EventPublisher | None = field(default_factory=StructlogEventPublisher)

    registry: TorchModelRegistry = field(default_factory=TorchModelRegistry)
    materializer: TensorMaterializer = field(
        default_factory=TorchTensorMaterializer
    )
    image_materializer: ImageTensorMaterializer = field(
        default_factory=PillowImageTensorMaterializer
    )
    profiler: ArchitectureProfiler = field(
        default_factory=TorchArchitectureProfiler
    )
    image_inspector: ImageFolderInspector = field(
        default_factory=PillowImageFolderInspector
    )
    baseline_detector: BaselineDetector = field(
        default_factory=StatisticalAnomalyDetector
    )
    trainer: ModelTrainer | None = None
    image_trainer: ImageModelTrainer | None = None
    evaluator: ModelEvaluator | None = None

    def __post_init__(self) -> None:
        # trainer 와 evaluator 는 같은 registry 를 공유해야 한다.
        # 학습이 남긴 모델을 평가가 찾을 수 있어야 하기 때문이다.
        if self.trainer is None:
            self.trainer = PyTorchModelTrainer(self.registry)
        if self.image_trainer is None:
            self.image_trainer = PyTorchImageTrainer(self.registry)
        if self.evaluator is None:
            self.evaluator = TorchModelEvaluator(self.registry)

    # -- Use Case ---------------------------------------------------------
    def compare_experiments(self) -> CompareExperiments:
        return CompareExperiments(self.runs)

    def compare_with_baseline(self) -> CompareWithBaseline:
        return CompareWithBaseline(self.runs, self.baseline_detector)

    def prepare_image_training_run(self) -> PrepareImageTrainingRun:
        return PrepareImageTrainingRun(
            self.runs,
            self.image_inspector,
            self.image_materializer,
            self.profiler,
            self.publisher,
        )

    def execute_image_training_run(self) -> ExecuteImageTrainingRun:
        return ExecuteImageTrainingRun(self.runs, self.image_trainer, self.publisher)

    def prepare_training_run(self) -> PrepareTrainingRun:
        return PrepareTrainingRun(
            self.runs,
            self.datasets,
            self.assessments,
            self.materializer,
            self.profiler,
            self.publisher,
        )

    def execute_training_run(self) -> ExecuteTrainingRun:
        return ExecuteTrainingRun(self.runs, self.trainer, self.publisher)

    def evaluate_model(self) -> EvaluateModel:
        return EvaluateModel(self.runs, self.evaluator, self.publisher)

    def evaluate_model_on_field(self) -> EvaluateModelOnField:
        return EvaluateModelOnField(self.runs, self.evaluator, self.publisher)

    def accept_model(self) -> AcceptModel:
        return AcceptModel(self.runs, self.publisher)

    def reopen_training_run(self) -> ReopenTrainingRun:
        return ReopenTrainingRun(self.runs, self.publisher)

    def get_training_run(self) -> GetTrainingRun:
        return GetTrainingRun(self.runs)

    def get_training_curve(self) -> GetTrainingCurve:
        return GetTrainingCurve(self.runs)

    def list_training_runs(self) -> ListTrainingRuns:
        return ListTrainingRuns(self.runs)


# =====================================================================
# Optimization Context (모듈 4)
# =====================================================================

import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

from application.optimization.benchmark_baseline import BenchmarkBaseline  # noqa: E402
from application.optimization.compare_candidates import (  # noqa: E402
    CompareCandidates,
    InspectArtifactSizes,
)
from application.optimization.compare_quantization import (  # noqa: E402
    CompareQuantization,
)
from application.optimization.convert_model import ConvertModel  # noqa: E402
from application.optimization.measure_resources import (  # noqa: E402
    MeasureResources,
    ScaleBatch,
)
from application.optimization.get_optimization_run import (  # noqa: E402
    GetOptimizationCertificate,
    GetOptimizationRun,
    GetRooflineProfile,
    ListOptimizationRuns,
)
from application.optimization.profile_roofline import ProfileRoofline
from application.optimization.reduce_structure import ReduceStructure  # noqa: E402
from application.optimization.select_model import (  # noqa: E402
    ReopenOptimizationRun,
    SelectModel,
)
from application.optimization.start_optimization_run import (  # noqa: E402
    StartOptimizationRun,
)
from domain.optimization.ports import (  # noqa: E402
    ArtifactAccuracyEvaluator,
    EquivalenceChecker,
    ModelExporter,
    OptimizationRunRepository,
    BatchScalingMeter,
    QuantizationLab,
    ResourceMeter,
    RooflineProfiler,
    StructuralReducer,
    RuntimeBenchmarker,
)
from infrastructure.optimization.exporters import (  # noqa: E402
    CompositeModelExporter,
    OnnxExporter,
    PyTorchExporter,
    TFLiteExporter,
    TorchScriptExporter,
)
from infrastructure.optimization.resource_meter import (  # noqa: E402
    BatchScalingBenchmarker,
    ProcessResourceMeter,
)
from infrastructure.optimization.quantize_aware import (  # noqa: E402
    TorchQuantizationLab,
)
from infrastructure.optimization.structural import (  # noqa: E402
    TorchStructuralReducer,
)
from infrastructure.optimization.measurement import (  # noqa: E402
    ArtifactAccuracyMeter,
    OutputEquivalenceChecker,
    RuntimeLatencyBenchmarker,
    TorchRooflineProfiler,
)
from infrastructure.optimization.runtime_registry import RuntimeRegistry  # noqa: E402
from infrastructure.persistence.in_memory_optimization_run_repository import (  # noqa: E402
    InMemoryOptimizationRunRepository,
)


def _default_artifact_dir() -> Path:
    return Path(tempfile.gettempdir()) / "on-device-ai" / "artifacts"


@dataclass
class OptimizationContainer:
    """Optimization Context 의 조립품.

    TrainingRunRepository 와 TorchModelRegistry 를 **모듈 3 과 공유해야 한다.**
    학습이 남긴 모델을 여기서 꺼내 변환하기 때문이다.
    따로 만들면 "학습된 모델을 찾을 수 없다"가 된다.

    tensorflow / onnxruntime / ai_edge_litert 라는 단어가 등장하는 곳은
    여기와 infrastructure/optimization 뿐이다.
    """

    training_runs: TrainingRunRepository = field(
        default_factory=InMemoryTrainingRunRepository
    )
    registry: TorchModelRegistry = field(default_factory=TorchModelRegistry)
    runs: OptimizationRunRepository = field(
        default_factory=InMemoryOptimizationRunRepository
    )
    publisher: EventPublisher | None = field(default_factory=StructlogEventPublisher)

    artifact_dir: Path = field(default_factory=_default_artifact_dir)
    runtimes: RuntimeRegistry = field(default_factory=RuntimeRegistry)

    exporter: ModelExporter | None = None
    checker: EquivalenceChecker | None = None
    benchmarker: RuntimeBenchmarker | None = None
    accuracy: ArtifactAccuracyEvaluator | None = None
    roofline: RooflineProfiler | None = None
    reducer: StructuralReducer | None = None
    quantization_lab: QuantizationLab | None = None
    resource_meter: ResourceMeter | None = None
    batch_meter: BatchScalingMeter | None = None

    def __post_init__(self) -> None:
        if self.exporter is None:
            self.exporter = CompositeModelExporter(
                [
                    PyTorchExporter(self.registry, self.runtimes, self.artifact_dir),
                    TorchScriptExporter(self.registry, self.runtimes, self.artifact_dir),
                    OnnxExporter(self.registry, self.runtimes, self.artifact_dir),
                    TFLiteExporter(self.registry, self.runtimes, self.artifact_dir),
                ]
            )
        if self.checker is None:
            self.checker = OutputEquivalenceChecker(self.registry, self.runtimes)
        if self.benchmarker is None:
            self.benchmarker = RuntimeLatencyBenchmarker(self.runtimes)
        if self.accuracy is None:
            self.accuracy = ArtifactAccuracyMeter(self.registry, self.runtimes)
        if self.resource_meter is None:
            self.resource_meter = ProcessResourceMeter(self.runtimes)
        if self.batch_meter is None:
            self.batch_meter = BatchScalingBenchmarker(self.runtimes)
        if self.quantization_lab is None:
            self.quantization_lab = TorchQuantizationLab(self.registry)
        if self.reducer is None:
            self.reducer = TorchStructuralReducer(self.registry, self.artifact_dir)
        if self.roofline is None:
            self.roofline = TorchRooflineProfiler(self.registry)

    @classmethod
    def sharing(cls, model_container: ModelContainer, **kwargs) -> OptimizationContainer:  # noqa: ANN003
        """모듈 3 컨테이너와 저장소를 공유하는 조립품을 만든다."""
        return cls(
            training_runs=model_container.runs,
            registry=model_container.registry,
            publisher=model_container.publisher,
            **kwargs,
        )

    # -- Use Case ---------------------------------------------------------
    def measure_resources(self) -> MeasureResources:
        return MeasureResources(self.runs, self.resource_meter)

    def scale_batch(self) -> ScaleBatch:
        return ScaleBatch(self.runs, self.batch_meter)

    def compare_quantization(self) -> CompareQuantization:
        return CompareQuantization(self.runs, self.quantization_lab)

    def reduce_structure(self) -> ReduceStructure:
        return ReduceStructure(self.runs, self.reducer, self.publisher)

    def start_optimization_run(self) -> StartOptimizationRun:
        return StartOptimizationRun(self.runs, self.training_runs, self.publisher)

    def benchmark_baseline(self) -> BenchmarkBaseline:
        return BenchmarkBaseline(
            self.runs, self.exporter, self.benchmarker, self.accuracy, self.publisher
        )

    def convert_model(self) -> ConvertModel:
        return ConvertModel(
            self.runs,
            self.exporter,
            self.checker,
            self.benchmarker,
            self.accuracy,
            self.publisher,
        )

    def profile_roofline(self) -> ProfileRoofline:
        return ProfileRoofline(self.runs, self.roofline, self.publisher)

    def compare_candidates(self) -> CompareCandidates:
        return CompareCandidates(self.runs)

    def inspect_artifact_sizes(self) -> InspectArtifactSizes:
        return InspectArtifactSizes(self.runs)

    def select_model(self) -> SelectModel:
        return SelectModel(self.runs, self.publisher)

    def reopen_optimization_run(self) -> ReopenOptimizationRun:
        return ReopenOptimizationRun(self.runs, self.publisher)

    def get_optimization_run(self) -> GetOptimizationRun:
        return GetOptimizationRun(self.runs)

    def get_optimization_certificate(self) -> GetOptimizationCertificate:
        return GetOptimizationCertificate(self.runs)

    def get_roofline_profile(self) -> GetRooflineProfile:
        return GetRooflineProfile(self.runs)

    def list_optimization_runs(self) -> ListOptimizationRuns:
        return ListOptimizationRuns(self.runs)


# =====================================================================
# Operations Context (모듈 5)
# =====================================================================

from application.operations.compare_shadow import CompareShadow  # noqa: E402
from application.operations.decide_retraining import DecideRetraining  # noqa: E402
from application.operations.deploy_model import (  # noqa: E402
    DeployModel,
    GetDeployment,
    ListDeployments,
    ReleaseVersion,
)
from application.operations.find_onset import (  # noqa: E402
    FindOnset,
    GetTimeline,
    GetWatch,
    GetWatchForDeployment,
    ListWatches,
)
from application.operations.observe_health import (  # noqa: E402
    IngestInferenceLog,
    ObserveHealth,
    RebaselineWatch,
)
from application.operations.respond_to_incident import (  # noqa: E402
    ListIncidents,
    QuarantineDeployment,
    ResolveIncident,
    ResumeDeployment,
    RollbackDeployment,
)
from application.shared.ports import Clock  # noqa: E402
from domain.operations.ports import (  # noqa: E402
    DeploymentRepository,
    HealthWatchRepository,
    InputDriftMeasurer,
    LatencyMeasurer,
    PredictionMixMeasurer,
    ShadowRunner,
)
from infrastructure.monitoring.event_log import SystemClock  # noqa: E402
from infrastructure.monitoring.field_measurers import (  # noqa: E402
    LogLatencyMeasurer,
    LogPredictionMixMeasurer,
)
from infrastructure.monitoring.inference_log_store import (  # noqa: E402
    InMemoryInferenceLogStore,
)
from infrastructure.persistence.in_memory_deployment_repository import (  # noqa: E402
    InMemoryDeploymentRepository,
    InMemoryHealthWatchRepository,
)


@dataclass
class OperationsContainer:
    """Operations Context 의 조립품.

    OptimizationRunRepository 와 TrainingRunRepository 를 **모듈 3·4 와 공유한다.**
    배포할 결과물과 그 전처리 통계를 거기서 가져오기 때문이다.

    입력 드리프트 측정기(InputDriftMeasurer)와 그림자 실행기(ShadowRunner)는
    기본값이 없다. **학습 분포와 실제 런타임이 있어야 만들 수 있기 때문이다** —
    없으면 없는 대로 두고, Use Case 쪽에서 측정을 건너뛴다.
    """

    optimization_runs: OptimizationRunRepository = field(
        default_factory=InMemoryOptimizationRunRepository
    )
    training_runs: TrainingRunRepository = field(
        default_factory=InMemoryTrainingRunRepository
    )
    deployments: DeploymentRepository = field(
        default_factory=InMemoryDeploymentRepository
    )
    watches: HealthWatchRepository = field(
        default_factory=InMemoryHealthWatchRepository
    )
    logs: InMemoryInferenceLogStore = field(default_factory=InMemoryInferenceLogStore)
    clock: Clock = field(default_factory=SystemClock)
    publisher: EventPublisher | None = field(default_factory=StructlogEventPublisher)

    latency: LatencyMeasurer | None = None
    mix: PredictionMixMeasurer | None = None
    drift: InputDriftMeasurer | None = None
    shadow: ShadowRunner | None = None

    def __post_init__(self) -> None:
        if self.latency is None:
            self.latency = LogLatencyMeasurer(self.logs)
        if self.mix is None:
            self.mix = LogPredictionMixMeasurer(self.logs)

    @classmethod
    def sharing(
        cls, optimization: OptimizationContainer, **kwargs
    ) -> OperationsContainer:  # noqa: ANN003
        """모듈 4 컨테이너와 저장소를 공유하는 조립품을 만든다."""
        return cls(
            optimization_runs=optimization.runs,
            training_runs=optimization.training_runs,
            publisher=optimization.publisher,
            **kwargs,
        )

    # -- Use Case ---------------------------------------------------------
    def deploy_model(self) -> DeployModel:
        return DeployModel(
            self.deployments,
            self.watches,
            self.optimization_runs,
            self.training_runs,
            self.clock,
            self.publisher,
        )

    def get_deployment(self) -> GetDeployment:
        return GetDeployment(self.deployments)

    def list_deployments(self) -> ListDeployments:
        return ListDeployments(self.deployments)

    def release_version(self) -> ReleaseVersion:
        return ReleaseVersion(
            self.deployments,
            self.optimization_runs,
            self.training_runs,
            self.clock,
            self.publisher,
        )

    def ingest_inference_log(self) -> IngestInferenceLog:
        return IngestInferenceLog(self.deployments, self.logs)

    def observe_health(self) -> ObserveHealth:
        return ObserveHealth(
            self.deployments,
            self.watches,
            self.logs,
            self.latency,
            self.mix,
            self.drift or _NullDriftMeasurer(),
            self.publisher,
        )

    def rebaseline_watch(self) -> RebaselineWatch:
        return RebaselineWatch(
            self.deployments, self.watches, self.mix, self.publisher
        )

    def find_onset(self) -> FindOnset:
        return FindOnset(self.watches)

    def get_timeline(self) -> GetTimeline:
        return GetTimeline(self.watches)

    def get_watch(self) -> GetWatch:
        return GetWatch(self.watches)

    def get_watch_for_deployment(self) -> GetWatchForDeployment:
        return GetWatchForDeployment(self.deployments, self.watches)

    def list_watches(self) -> ListWatches:
        return ListWatches(self.watches)

    def quarantine_deployment(self) -> QuarantineDeployment:
        return QuarantineDeployment(
            self.deployments, self.watches, self.clock, self.publisher
        )

    def resume_deployment(self) -> ResumeDeployment:
        return ResumeDeployment(self.deployments, self.clock, self.publisher)

    def rollback_deployment(self) -> RollbackDeployment:
        return RollbackDeployment(self.deployments, self.clock, self.publisher)

    def resolve_incident(self) -> ResolveIncident:
        return ResolveIncident(self.watches, self.publisher)

    def list_incidents(self) -> ListIncidents:
        return ListIncidents(self.watches)

    def compare_shadow(self) -> CompareShadow:
        if self.shadow is None:
            from application.shared.errors import UnsupportedOperation

            raise UnsupportedOperation(
                "그림자 실행기가 조립되지 않았다. "
                "비교하려면 새 결과물을 실제로 돌릴 수 있어야 한다.",
                subject="shadow",
            )
        return CompareShadow(self.deployments, self.shadow)

    def decide_retraining(self) -> DecideRetraining:
        return DecideRetraining(self.watches, self.logs, self.publisher)


class _NullDriftMeasurer:
    """학습 분포를 안 넘겨줬을 때 쓰는 자리 지킴이.

    빈 보고서를 돌려주고, DriftPolicy 가 "기준이 없다"고 지적한다.
    **아무 말 없이 통과시키지 않는다.**
    """

    def measure(self, deployment_id, window):  # noqa: ANN001, ANN201
        from domain.operations.drift import DriftReport

        return DriftReport(window=window, features=())


# =====================================================================
# Fleet Context (모듈 6)
# =====================================================================

from application.fleet.build_and_train import (  # noqa: E402
    BuildTrainingDataset,
    PollTrainingJob,
    StopTrainingJob,
    SubmitTrainingJob,
)
from application.fleet.govern_storage import (  # noqa: E402
    DeployEndpoint,
    GovernStorage,
    InspectStorage,
    RecordExperiment,
    ReviewExperiment,
    TeardownEndpoint,
)
from application.fleet.ingest_uplink import (  # noqa: E402
    CreateFleet,
    IngestUplink,
    InspectLakeLayout,
    MarkDevice,
    SummarizeFleet,
    SweepStaleDevices,
)
from application.fleet.release_and_rollout import (  # noqa: E402
    AdvanceRollout,
    ApplyRolloutOutcomes,
    CollectWave,
    GetRollout,
    HaltRollout,
    ListRollouts,
    PlanRollout,
    PromoteRelease,
    PublishRelease,
    RollbackRollout,
    TraceLineage,
)
from domain.fleet.ports import (  # noqa: E402
    BucketGovernanceGateway,
    EndpointGateway,
    ExperimentStore,
    DeviceRegistry,
    FleetRepository,
    ObjectStore,
    OtaGateway,
    RemoteTrainingGateway,
    RolloutRepository,
)
from infrastructure.aws.config import AwsConfig  # noqa: E402
from infrastructure.persistence.in_memory_fleet_repository import (  # noqa: E402
    InMemoryFleetRepository,
    InMemoryRolloutRepository,
)


@dataclass
class FleetContainer:
    """Fleet Context 의 조립품.

    **AWS 어댑터를 여기서만 만든다.** (CLAUDE.md §15, §20)
    Application 도 Domain 도 자기가 S3 위에서 도는지 모른다.

    네 어댑터는 기본값이 없다 — AWS 자격증명이 없는 환경에서 컨테이너를 만드는 것만으로
    boto3 세션이 열리면 안 되기 때문이다. `with_aws()` 로 명시적으로 붙인다.
    """

    fleets: FleetRepository = field(default_factory=InMemoryFleetRepository)
    rollouts: RolloutRepository = field(default_factory=InMemoryRolloutRepository)
    clock: Clock = field(default_factory=SystemClock)
    publisher: EventPublisher | None = field(default_factory=StructlogEventPublisher)

    store: ObjectStore | None = None
    registry: DeviceRegistry | None = None
    training: RemoteTrainingGateway | None = None
    ota: OtaGateway | None = None
    experiments: ExperimentStore | None = None
    governance: BucketGovernanceGateway | None = None
    endpoints: EndpointGateway | None = None
    aws: AwsConfig = field(default_factory=AwsConfig)

    @classmethod
    def with_aws(cls, config: AwsConfig | None = None, **kwargs) -> FleetContainer:  # noqa: ANN003
        """AWS 어댑터를 붙여 조립한다.

        **이 클래스 메서드가 이 프로젝트에서 boto3 가 실제로 깨어나는 유일한 지점이다.**
        """
        from infrastructure.aws.dynamo_device_registry import DynamoDeviceRegistry
        from infrastructure.aws.iot_ota_gateway import IotJobsOtaGateway
        from infrastructure.aws.s3_object_store import S3ObjectStore
        from infrastructure.aws.s3_experiment_store import S3ExperimentStore
        from infrastructure.aws.s3_governance import S3Governance
        from infrastructure.aws.sagemaker_endpoint import SageMakerEndpointGateway
        from infrastructure.aws.sagemaker_training import SageMakerTrainingGateway

        settings = config or AwsConfig()
        store = S3ObjectStore(settings)
        registry = DynamoDeviceRegistry(settings)
        store.ensure_bucket()
        registry.ensure_tables()

        return cls(
            aws=settings,
            store=store,
            registry=registry,
            training=SageMakerTrainingGateway(settings),
            ota=IotJobsOtaGateway(settings),
            experiments=S3ExperimentStore(settings),
            governance=S3Governance(settings),
            endpoints=SageMakerEndpointGateway(settings),
            **kwargs,
        )

    # -- Use Case ---------------------------------------------------------
    def record_experiment(self) -> RecordExperiment:
        return RecordExperiment(self._experiments())

    def review_experiment(self) -> ReviewExperiment:
        return ReviewExperiment(self._experiments())

    def govern_storage(self) -> GovernStorage:
        return GovernStorage(self._governance())

    def inspect_storage(self) -> InspectStorage:
        return InspectStorage(self._governance())

    def deploy_endpoint(self) -> DeployEndpoint:
        return DeployEndpoint(self._endpoints())

    def teardown_endpoint(self) -> TeardownEndpoint:
        return TeardownEndpoint(self._endpoints())

    def create_fleet(self) -> CreateFleet:
        return CreateFleet(self.fleets, self._registry(), self.publisher)

    def ingest_uplink(self) -> IngestUplink:
        return IngestUplink(
            self.fleets, self._store(), self._registry(), self.publisher
        )

    def inspect_lake_layout(self) -> InspectLakeLayout:
        return InspectLakeLayout(self.fleets, self._store())

    def summarize_fleet(self) -> SummarizeFleet:
        return SummarizeFleet(self.fleets)

    def sweep_stale_devices(self) -> SweepStaleDevices:
        return SweepStaleDevices(self.fleets, self._registry(), self.publisher)

    def mark_device(self) -> MarkDevice:
        return MarkDevice(self.fleets, self._registry(), self.publisher)

    def build_training_dataset(self) -> BuildTrainingDataset:
        return BuildTrainingDataset(self.fleets, self._store(), self._registry())

    def submit_training_job(self) -> SubmitTrainingJob:
        return SubmitTrainingJob(self._training(), self.publisher)

    def poll_training_job(self) -> PollTrainingJob:
        return PollTrainingJob(self._training(), self.publisher)

    def stop_training_job(self) -> StopTrainingJob:
        return StopTrainingJob(self._training())

    def publish_release(self) -> PublishRelease:
        return PublishRelease(self.fleets, self.publisher)

    def promote_release(self) -> PromoteRelease:
        return PromoteRelease(self.fleets, self.publisher)

    def plan_rollout(self) -> PlanRollout:
        return PlanRollout(
            self.fleets, self.rollouts, self._ota(), self.clock, self.publisher
        )

    def collect_wave(self) -> CollectWave:
        return CollectWave(self.rollouts, self._ota(), self.clock, self.publisher)

    def advance_rollout(self) -> AdvanceRollout:
        return AdvanceRollout(
            self.fleets, self.rollouts, self._ota(), self.clock, self.publisher
        )

    def halt_rollout(self) -> HaltRollout:
        return HaltRollout(self.rollouts, self.clock, self.publisher)

    def rollback_rollout(self) -> RollbackRollout:
        return RollbackRollout(
            self.fleets, self.rollouts, self._ota(), self.clock, self.publisher
        )

    def apply_rollout_outcomes(self) -> ApplyRolloutOutcomes:
        return ApplyRolloutOutcomes(
            self.fleets, self.rollouts, self._registry(), self.publisher
        )

    def trace_lineage(self) -> TraceLineage:
        return TraceLineage(self.fleets)

    def get_rollout(self) -> GetRollout:
        return GetRollout(self.rollouts)

    def list_rollouts(self) -> ListRollouts:
        return ListRollouts(self.rollouts)

    # -- 내부 --------------------------------------------------------------
    def _store(self) -> ObjectStore:
        return _required(self.store, "객체 저장소")

    def _registry(self) -> DeviceRegistry:
        return _required(self.registry, "디바이스 레지스트리")

    def _training(self) -> RemoteTrainingGateway:
        return _required(self.training, "원격 학습 게이트웨이")

    def _ota(self) -> OtaGateway:
        return _required(self.ota, "OTA 게이트웨이")

    def _experiments(self) -> ExperimentStore:
        return _required(self.experiments, "실험 기록 저장소")

    def _governance(self) -> BucketGovernanceGateway:
        return _required(self.governance, "저장소 거버넌스 게이트웨이")

    def _endpoints(self) -> EndpointGateway:
        return _required(self.endpoints, "엔드포인트 게이트웨이")


def _required(adapter, name: str):  # noqa: ANN001, ANN201
    if adapter is None:
        from application.shared.errors import UnsupportedOperation

        raise UnsupportedOperation(
            f"{name} 가 조립되지 않았다. "
            "FleetContainer.with_aws() 로 붙이거나 직접 넣는다.",
            subject=name,
        )
    return adapter
