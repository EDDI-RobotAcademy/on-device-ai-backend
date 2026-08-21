"""컨테이너 주입.

테스트에서는 app.dependency_overrides 로 갈아끼운다.
전역 상태를 숨기지 않기 위해 모듈 수준 싱글턴 하나만 두고, 그것도 교체 가능하게 한다.

Data Quality Container 는 Dataset 저장소를 Data Container 와 **공유한다.**
품질 평가는 Dataset 을 번역해서 시작하므로, 두 Context 가 같은 저장소를 봐야 한다.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from infrastructure.config.container import (
    DataContainer,
    DataQualityContainer,
    FleetContainer,
    ModelContainer,
    OperationsContainer,
    OptimizationContainer,
)

_container: DataContainer | None = None
_quality_container: DataQualityContainer | None = None
_model_container: ModelContainer | None = None
_optimization_container: OptimizationContainer | None = None
_operations_container: OperationsContainer | None = None
_fleet_container: FleetContainer | None = None


def get_container() -> DataContainer:
    global _container
    if _container is None:
        _container = DataContainer()
    return _container


def get_quality_container() -> DataQualityContainer:
    global _quality_container
    if _quality_container is None:
        _quality_container = DataQualityContainer(
            datasets=get_container().repository,
            publisher=get_container().publisher,
        )
    return _quality_container


def get_model_container() -> ModelContainer:
    global _model_container
    if _model_container is None:
        _model_container = ModelContainer(
            datasets=get_container().repository,
            assessments=get_quality_container().assessments,
            publisher=get_container().publisher,
        )
    return _model_container


def get_optimization_container() -> OptimizationContainer:
    """Model Container 와 **저장소를 공유한다.**

    최적화는 학습이 남긴 모델 그 자체를 변환한다.
    따로 만들면 "학습된 모델을 찾을 수 없다"가 된다.
    """
    global _optimization_container
    if _optimization_container is None:
        _optimization_container = OptimizationContainer.sharing(
            get_model_container()
        )
    return _optimization_container


def get_operations_container() -> OperationsContainer:
    """Optimization Container 와 저장소를 공유한다.

    배포할 결과물과 그 전처리 통계를 거기서 가져오기 때문이다.
    """
    global _operations_container
    if _operations_container is None:
        _operations_container = OperationsContainer.sharing(
            get_optimization_container()
        )
    return _operations_container


def get_fleet_container() -> FleetContainer:
    """**AWS 어댑터는 붙이지 않는다.**

    자격증명이 없는 환경에서 서버를 띄우는 것만으로 boto3 세션이 열리면 안 된다.
    실제로 쓰려면 `set_fleet_container(FleetContainer.with_aws(...))` 로 갈아끼운다.
    """
    global _fleet_container
    if _fleet_container is None:
        _fleet_container = FleetContainer()
    return _fleet_container


def set_container(container: DataContainer | None) -> None:
    """테스트/실습에서 조립품을 통째로 바꾼다."""
    global _container
    _container = container


def set_quality_container(container: DataQualityContainer | None) -> None:
    global _quality_container
    _quality_container = container


def set_model_container(container: ModelContainer | None) -> None:
    global _model_container
    _model_container = container


def set_optimization_container(container: OptimizationContainer | None) -> None:
    global _optimization_container
    _optimization_container = container


def set_operations_container(container: OperationsContainer | None) -> None:
    global _operations_container
    _operations_container = container


def set_fleet_container(container: FleetContainer | None) -> None:
    global _fleet_container
    _fleet_container = container


container_dependency = Annotated[DataContainer, Depends(get_container)]
quality_container_dependency = Annotated[
    DataQualityContainer, Depends(get_quality_container)
]
model_container_dependency = Annotated[ModelContainer, Depends(get_model_container)]
optimization_container_dependency = Annotated[
    OptimizationContainer, Depends(get_optimization_container)
]
operations_container_dependency = Annotated[
    OperationsContainer, Depends(get_operations_container)
]
fleet_container_dependency = Annotated[FleetContainer, Depends(get_fleet_container)]
