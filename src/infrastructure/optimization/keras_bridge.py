"""PyTorch 모델을 TensorFlow 로 옮긴다. (실습 4-4)

**이 파일이 CLAUDE.md §14 의 값어치를 증명하는 곳이다.**

모듈 3 에서 `ModelArchitecture` 는 "어떤 모양의 계산을 할 것인가"만 담았다.
PyTorch 의 nn.Module 이 아니라 **명세**였다.

그래서 같은 명세로 Keras 모델을 조립할 수 있다.
가중치만 옮기면 두 프레임워크가 같은 답을 낸다.

옮길 때 반드시 바꿔야 하는 것: **축 순서**

    PyTorch Conv1d 커널   (out_channels, in_channels, kernel)
    Keras   Conv1D 커널   (kernel, in_channels, out_channels)

    PyTorch Linear 가중치 (out_features, in_features)
    Keras   Dense  가중치 (in_features, out_features)

실습 3-3 에서 "축을 틀려도 reshape 이 그냥 된다"고 했던 그 문제가
여기서 실제 비용으로 돌아온다. 그래서 옮긴 뒤에 반드시 대조한다.
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from domain.model.architecture import ArchitectureKind, ModelArchitecture  # noqa: E402
from infrastructure.errors import UnsupportedSourceFormat  # noqa: E402


def build_keras_model(architecture: ModelArchitecture, state_dict: dict):  # noqa: ANN201
    """명세로부터 Keras 모델을 조립하고 PyTorch 가중치를 이식한다."""
    import tensorflow as tf

    if architecture.kind is ArchitectureKind.CNN1D:
        return _build_cnn1d(architecture, state_dict, tf)
    if architecture.kind is ArchitectureKind.MLP:
        return _build_mlp(architecture, state_dict, tf)
    raise UnsupportedSourceFormat(
        f"{architecture.kind.value} 는 아직 TFLite 경로를 지원하지 않는다. "
        "지원하지 않는다는 사실을 조용히 숨기지 않는다.",
        subject=architecture.kind.value,
    )


def _build_cnn1d(architecture: ModelArchitecture, state_dict: dict, tf):  # noqa: ANN001, ANN202
    length, channels = architecture.input_spec.shape
    inputs = tf.keras.Input(shape=(length, channels))

    x = inputs
    conv_layers = []
    for out_channels in architecture.hidden_channels:
        conv = tf.keras.layers.Conv1D(
            out_channels, architecture.kernel_size, padding="same"
        )
        x = tf.keras.layers.ReLU()(conv(x))
        conv_layers.append(conv)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    dense = tf.keras.layers.Dense(architecture.class_count)
    model = tf.keras.Model(inputs, dense(x))

    # PyTorch 의 features.0, features.2, ... 는 Conv1d 이고 그 사이는 ReLU 다.
    for position, conv in enumerate(conv_layers):
        weight = state_dict[f"features.{position * 2}.weight"].numpy()
        bias = state_dict[f"features.{position * 2}.bias"].numpy()
        conv.set_weights([weight.transpose(2, 1, 0), bias])  # (O,I,K) → (K,I,O)

    dense.set_weights(
        [
            state_dict["classifier.weight"].numpy().T,  # (O,I) → (I,O)
            state_dict["classifier.bias"].numpy(),
        ]
    )
    return model


def _build_mlp(architecture: ModelArchitecture, state_dict: dict, tf):  # noqa: ANN001, ANN202
    length, channels = architecture.input_spec.shape
    inputs = tf.keras.Input(shape=(length, channels))
    x = tf.keras.layers.Flatten()(inputs)

    dense_layers = []
    for out_features in architecture.hidden_channels:
        dense = tf.keras.layers.Dense(out_features)
        x = tf.keras.layers.ReLU()(dense(x))
        dense_layers.append(dense)
    final = tf.keras.layers.Dense(architecture.class_count)
    model = tf.keras.Model(inputs, final(x))

    # PyTorch: Flatten, Linear, ReLU, Linear, ReLU, ..., Dropout, Linear
    torch_indices = [1 + i * 2 for i in range(len(dense_layers))]
    for dense, index in zip(dense_layers, torch_indices, strict=True):
        dense.set_weights(
            [
                state_dict[f"net.{index}.weight"].numpy().T,
                state_dict[f"net.{index}.bias"].numpy(),
            ]
        )
    last = torch_indices[-1] + 3  # ReLU 하나 + Dropout 하나를 건너뛴다
    final.set_weights(
        [state_dict[f"net.{last}.weight"].numpy().T, state_dict[f"net.{last}.bias"].numpy()]
    )
    return model


def keras_predict(model, x: np.ndarray) -> np.ndarray:  # noqa: ANN001
    return np.asarray(model(x, training=False))
