"""실습용 현장 데이터 생성기.

실제 기업 데이터를 교재에 그대로 실을 수 없으므로, 현장에서 실제로 마주치는
결함을 **의도적으로 심은** 합성 데이터를 만든다.

심어 놓은 결함 목록은 각 생성 모듈의 docstring 에 전부 적혀 있다.
정답을 알고 있는 데이터로 연습해야, 정답을 모르는 데이터에서 같은 것을 찾을 수 있다.

    plant_power.py          모듈 1 — **구조**가 깨진 데이터 (D01~D13)
    plant_power_quality.py  모듈 2 — 구조는 멀쩡하고 **내용**이 오염된 데이터 (Q01~Q10)
    plant_power_model.py    모듈 3 — 두 게이트를 통과한 학습용 데이터 + 현장 홀드아웃
    plant_power_operations.py 모듈 5 — **배포 뒤 4일치 현장 신호** (O01~O06)
    casting_images.py       모듈 1 — 이미지 결함 (I01~I05)
    industrial_images.py    모듈 3 — **학습이 되는** 산업 이미지 (다이캐스팅 / 식품)

seed 를 고정하므로 몇 번을 돌려도 같은 파일이 나온다.
"""

from infrastructure.sample_data.casting_images import (
    CastingImageSample,
    write_casting_images,
)
from infrastructure.sample_data.industrial_images import (
    IndustrialImageSample,
    write_industrial_images,
)
from infrastructure.sample_data.plant_power import (
    PlantPowerSample,
    write_plant_power_samples,
)
from infrastructure.sample_data.plant_power_model import (
    ModelSample,
    write_model_samples,
)
from infrastructure.sample_data.plant_power_operations import (
    OperationsSample,
    write_operations_samples,
)
from infrastructure.sample_data.plant_power_quality import (
    QualitySample,
    write_quality_samples,
)

__all__ = [
    "CastingImageSample",
    "IndustrialImageSample",
    "ModelSample",
    "OperationsSample",
    "PlantPowerSample",
    "QualitySample",
    "write_casting_images",
    "write_industrial_images",
    "write_model_samples",
    "write_operations_samples",
    "write_plant_power_samples",
    "write_quality_samples",
]
