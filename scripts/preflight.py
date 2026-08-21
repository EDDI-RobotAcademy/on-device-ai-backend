"""설치 점검.

과정 전체에서 쓰는 라이브러리가 전부 들어와 있는지 처음에 한 번 확인한다.
실습 중간에 ImportError 로 멈추는 상황을 만들지 않기 위한 것이다.

    .venv/bin/python -m scripts.preflight
"""

from __future__ import annotations

import importlib
import sys

# (import 이름, 표시 이름, 이 과정의 어느 단계에서 쓰는가)
REQUIRED: tuple[tuple[str, str, str], ...] = (
    ("fastapi", "FastAPI", "API"),
    ("pydantic", "Pydantic", "API"),
    ("uvicorn", "Uvicorn", "API"),
    ("numpy", "NumPy", "데이터"),
    ("pandas", "pandas", "데이터"),
    ("pyarrow", "PyArrow", "데이터"),
    ("scipy", "SciPy", "데이터 품질"),
    ("sklearn", "scikit-learn", "데이터 품질"),
    ("PIL", "Pillow", "이미지"),
    ("torch", "PyTorch", "모델"),
    ("torchvision", "torchvision", "모델"),
    ("onnx", "ONNX", "최적화"),
    ("onnxscript", "onnxscript", "최적화(torch.onnx.export)"),
    ("psutil", "psutil", "최적화(실습 4-13 자원 실측)"),
    ("onnxruntime", "ONNX Runtime", "최적화"),
    ("tensorflow", "TensorFlow", "최적화(TFLite 변환)"),
    ("ai_edge_litert", "LiteRT", "최적화(TFLite 실행)"),
    ("sqlalchemy", "SQLAlchemy", "운영"),
    ("structlog", "structlog", "운영"),
    ("boto3", "boto3", "AWS"),
    ("botocore", "botocore", "AWS"),
    ("moto", "moto", "AWS 테스트 (S3/DynamoDB/SageMaker/IoT)"),
    ("pytest", "pytest", "테스트"),
    ("httpx", "httpx", "테스트"),
)


def main() -> int:
    print(f"Python {sys.version.split()[0]}  ({sys.executable})\n")
    missing: list[str] = []
    for module_name, display, stage in REQUIRED:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - 원인을 그대로 보여준다
            print(f"  ✗ {display:<16} {stage:<20} {type(exc).__name__}: {exc}")
            missing.append(display)
            continue
        version = getattr(module, "__version__", "?")
        print(f"  ✓ {display:<16} {stage:<20} {version}")

    if missing:
        print(f"\n설치되지 않은 것: {', '.join(missing)}")
        print("  .venv/bin/python -m pip install -r requirements.txt")
        return 1

    print("\n전부 준비됐다. 실습을 시작해도 된다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
