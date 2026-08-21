"""실행 경로와 정밀도를 바꾸는 어댑터. (실습 4-2 ~ 4-7)

내보내기(export)는 두 가지를 한다.
    1. 파일을 만든다 — 크기는 **파일에서 실제로 잰다**
    2. 그 파일을 실행할 수 있는 형태로 RuntimeRegistry 에 넣는다

지원하지 않는 조합은 `supports()` 가 False 를 돌려준다.
조용히 비슷한 것을 만들어 주지 않는다 — 그게 가장 나쁜 실패다.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from domain.model.architecture import ModelArchitecture
from domain.optimization.baseline_ref import BaselineModelRef
from domain.optimization.errors import ConversionFailed
from domain.optimization.identifiers import ArtifactId
from domain.optimization.runtime import ModelArtifact, Precision, RuntimeTarget
from infrastructure.ml.torch_trainer import TorchModelRegistry
from infrastructure.optimization.runtime_registry import LoadedRuntime, RuntimeRegistry

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


class _TorchBackedExporter:
    """모듈 3 이 학습해 둔 모델을 꺼내 쓰는 공통 부분."""

    def __init__(
        self,
        models: TorchModelRegistry,
        runtimes: RuntimeRegistry,
        artifact_dir: Path,
    ) -> None:
        self._models = models
        self._runtimes = runtimes
        self._dir = Path(artifact_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _trained(self, baseline: BaselineModelRef):  # noqa: ANN202
        from domain.model.identifiers import ModelVersionId

        trained = self._models.get(ModelVersionId.of(baseline.model_version_id))
        if trained is None:
            raise ConversionFailed(
                f"학습된 모델을 찾을 수 없다: {baseline.model_version_id}",
                subject=baseline.model_version_id,
            )
        module = trained.module
        module.load_state_dict(trained.best_state)  # 최고 지점 (실습 3-7)
        module.eval()
        return trained, module

    def _example(self, baseline: BaselineModelRef) -> np.ndarray:
        return np.zeros((1, *baseline.input_shape), dtype="float32")

    def _artifact(
        self,
        baseline: BaselineModelRef,
        runtime: RuntimeTarget,
        precision: Precision,
        path: Path,
    ) -> ModelArtifact:
        return ModelArtifact(
            artifact_id=ArtifactId.of(
                f"{baseline.model_version_id}-{runtime.value}-{precision.value}".lower()
            ),
            runtime=runtime,
            precision=precision,
            size_bytes=path.stat().st_size,
            uri=str(path),
            parameter_count=baseline.parameter_count,
        )

    def _activation_bytes(self, architecture: ModelArchitecture) -> int:
        """추론 중 잡아야 하는 중간 결과의 크기 (분석 추정).

        연속한 두 층의 출력이 동시에 살아 있다고 보는 이중 버퍼 근사다.
        실제 런타임은 더 아낄 수도, 더 쓸 수도 있다 — 그래서 '추정'이라고 적는다.
        """
        from infrastructure.ml.torch_architecture import TorchArchitectureProfiler

        profile = TorchArchitectureProfiler().profile(architecture)
        sizes = [1, *(_elements(layer.output_shape) for layer in profile.layers)]
        pairs = [sizes[i] + sizes[i + 1] for i in range(len(sizes) - 1)]
        inputs = _elements(architecture.input_spec.shape)
        return (max(pairs, default=0) + inputs) * 4


def _elements(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


class PyTorchExporter(_TorchBackedExporter):
    """기준 모델. 변환하지 않는다 — 비교의 출발점을 만든다. (실습 4-1)"""

    def supports(self, runtime: RuntimeTarget, precision: Precision) -> bool:
        return runtime is RuntimeTarget.PYTORCH and precision is Precision.FP32

    def export(
        self, baseline: BaselineModelRef, runtime: RuntimeTarget, precision: Precision
    ) -> ModelArtifact:
        trained, module = self._trained(baseline)
        path = self._dir / f"{baseline.model_version_id}-eager.pt"
        torch.save(module.state_dict(), path)

        artifact = self._artifact(baseline, runtime, precision, path)

        def predict(x: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return module(torch.from_numpy(np.ascontiguousarray(x))).numpy()

        self._runtimes.put(
            artifact.artifact_id,
            LoadedRuntime(
                predict=predict,
                input_shape=tuple(baseline.input_shape),
                activation_bytes=self._activation_bytes(trained.architecture),
                note="파이썬 인터프리터가 층마다 함수를 부른다",
            ),
        )
        return artifact


class TorchScriptExporter(_TorchBackedExporter):
    """파이썬을 벗어난 그래프로 굳힌다. (실습 4-2)

    `trace` 는 **한 번 실행해 보고 그 경로를 기록한다.**
    그래서 forward 안에 입력에 따라 갈리는 if 가 있으면 한쪽만 남는다.
    학습용 코드를 그대로 내보내면 안 되는 이유다.
    """

    def supports(self, runtime: RuntimeTarget, precision: Precision) -> bool:
        return runtime is RuntimeTarget.TORCHSCRIPT and precision is Precision.FP32

    def export(
        self, baseline: BaselineModelRef, runtime: RuntimeTarget, precision: Precision
    ) -> ModelArtifact:
        trained, module = self._trained(baseline)
        example = torch.from_numpy(self._example(baseline))

        try:
            scripted = torch.jit.trace(module, example)
            scripted = torch.jit.freeze(scripted)  # dropout/BN 을 상수로 굳힌다
        except Exception as exc:  # noqa: BLE001
            raise ConversionFailed(
                f"TorchScript 변환 실패: {exc}", subject="TORCHSCRIPT"
            ) from exc

        path = self._dir / f"{baseline.model_version_id}-scripted.pt"
        scripted.save(str(path))
        artifact = self._artifact(baseline, runtime, precision, path)

        def predict(x: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return scripted(torch.from_numpy(np.ascontiguousarray(x))).numpy()

        self._runtimes.put(
            artifact.artifact_id,
            LoadedRuntime(
                predict=predict,
                input_shape=tuple(baseline.input_shape),
                activation_bytes=self._activation_bytes(trained.architecture),
                note="파이썬 없이 실행되지만 PyTorch 런타임은 여전히 필요하다",
            ),
        )
        return artifact


class OnnxExporter(_TorchBackedExporter):
    """프레임워크 중립 그래프로 내보낸다. (실습 4-3)

    ONNX 는 파일 형식이자 연산자 규격이다.
    PyTorch 의 연산이 ONNX 연산자로 **치환**되므로, 그 과정에서 의미가 조금 바뀔 수 있다.
    그래서 내보낸 뒤 반드시 대조한다.
    """

    def __init__(self, *args, opset_version: int = 17, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self._opset = opset_version

    def supports(self, runtime: RuntimeTarget, precision: Precision) -> bool:
        return runtime is RuntimeTarget.ONNX and precision is Precision.FP32

    def export(
        self, baseline: BaselineModelRef, runtime: RuntimeTarget, precision: Precision
    ) -> ModelArtifact:
        import onnxruntime as ort

        trained, module = self._trained(baseline)
        path = self._dir / f"{baseline.model_version_id}.onnx"

        try:
            torch.onnx.export(
                module,
                (torch.from_numpy(self._example(baseline)),),
                str(path),
                input_names=["input"],
                output_names=["logits"],
                # 배치 축만 동적으로 둔다. 나머지가 동적이면 런타임이 최적화를 포기한다.
                dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=self._opset,
                dynamo=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise ConversionFailed(f"ONNX 변환 실패: {exc}", subject="ONNX") from exc

        artifact = self._artifact(baseline, runtime, precision, path)

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1  # 디바이스 조건에 맞춘다 (실습 4-1)
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )

        def predict(x: np.ndarray) -> np.ndarray:
            return session.run(None, {"input": np.ascontiguousarray(x)})[0]

        self._runtimes.put(
            artifact.artifact_id,
            LoadedRuntime(
                predict=predict,
                input_shape=tuple(baseline.input_shape),
                activation_bytes=self._activation_bytes(trained.architecture),
                note="여러 런타임이 같은 파일을 읽는다",
            ),
        )
        return artifact


class TFLiteExporter(_TorchBackedExporter):
    """임베디드용으로 내보낸다. (실습 4-4, 4-6, 4-7)

    경로: PyTorch → (명세로 재조립) Keras → TFLite

    직접 변환기가 없어서 우회하는 것이 아니다.
    **명세가 있으면 프레임워크를 바꿀 수 있다**는 것을 보이는 경로다.
    가중치를 옮긴 뒤 반드시 대조한다 — 축 순서를 틀리면 조용히 이상해진다.
    """

    def __init__(self, *args, representative_samples: int = 64, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self._representative_samples = representative_samples

    def supports(self, runtime: RuntimeTarget, precision: Precision) -> bool:
        return runtime is RuntimeTarget.TFLITE

    def export(
        self, baseline: BaselineModelRef, runtime: RuntimeTarget, precision: Precision
    ) -> ModelArtifact:
        import tensorflow as tf
        from ai_edge_litert.interpreter import Interpreter

        from infrastructure.optimization.keras_bridge import build_keras_model

        trained, module = self._trained(baseline)
        try:
            keras_model = build_keras_model(trained.architecture, module.state_dict())
            converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
            self._configure(converter, precision, trained, tf)
            blob = converter.convert()
        except Exception as exc:  # noqa: BLE001
            raise ConversionFailed(
                f"TFLite {precision.value} 변환 실패: {exc}", subject="TFLITE"
            ) from exc

        path = self._dir / f"{baseline.model_version_id}-{precision.value.lower()}.tflite"
        path.write_bytes(blob)
        artifact = self._artifact(baseline, runtime, precision, path)

        interpreter = Interpreter(model_content=blob, num_threads=1)
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]

        # get_input_details() 가 돌려준 dict 는 **그 시점의 사본**이다.
        # resize 해도 이 dict 는 안 바뀐다. 그래서 현재 모양을 따로 들고 있어야 한다.
        # (이걸 놓치면 "Got 1 but expected 66" 같은 오류가 뒤늦게 나온다)
        current_shape = tuple(input_detail["shape"])

        def predict(x: np.ndarray) -> np.ndarray:
            nonlocal current_shape
            batch = np.ascontiguousarray(x, dtype="float32")
            if current_shape != batch.shape:
                interpreter.resize_tensor_input(input_detail["index"], batch.shape)
                interpreter.allocate_tensors()
                current_shape = batch.shape
            interpreter.set_tensor(input_detail["index"], batch)
            interpreter.invoke()
            return np.array(interpreter.get_tensor(output_detail["index"]))

        self._runtimes.put(
            artifact.artifact_id,
            LoadedRuntime(
                predict=predict,
                input_shape=tuple(baseline.input_shape),
                activation_bytes=self._activation_bytes(trained.architecture),
                note="인터프리터가 작고 연산자가 제한되어 있다",
            ),
        )
        return artifact

    def _configure(self, converter, precision: Precision, trained, tf) -> None:  # noqa: ANN001
        if precision is Precision.FP32:
            return
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if precision is Precision.FP16:
            converter.target_spec.supported_types = [tf.float16]
            return

        # INT8 — 값의 범위를 알아야 정수 눈금을 정할 수 있다.
        # 그 범위는 **train 분할에서** 뽑는다. 실습 1-7 과 같은 이유다.
        samples = trained.dataset.features["train"]
        count = min(self._representative_samples, len(samples))
        if count == 0:
            raise ConversionFailed(
                "대표 데이터가 없다. INT8 양자화는 값의 범위를 모르면 할 수 없다.",
                subject="representative_dataset",
            )

        def representative():  # noqa: ANN202
            for row in samples[:count]:
                yield [row[None].astype("float32")]

        converter.representative_dataset = representative


class CompositeModelExporter:
    """domain.optimization.ports.ModelExporter 구현.

    조합마다 다른 어댑터가 맡는다. 어느 것도 못 맡으면 지원하지 않는다고 말한다.
    """

    def __init__(self, exporters: list) -> None:  # noqa: ANN001
        self._exporters = exporters

    def supports(self, runtime: RuntimeTarget, precision: Precision) -> bool:
        return any(e.supports(runtime, precision) for e in self._exporters)

    def export(
        self, baseline: BaselineModelRef, runtime: RuntimeTarget, precision: Precision
    ) -> ModelArtifact:
        for exporter in self._exporters:
            if exporter.supports(runtime, precision):
                return exporter.export(baseline, runtime, precision)
        raise ConversionFailed(
            f"{runtime.value}/{precision.value} 조합을 지원하는 어댑터가 없다.",
            subject=f"{runtime.value}/{precision.value}",
        )
