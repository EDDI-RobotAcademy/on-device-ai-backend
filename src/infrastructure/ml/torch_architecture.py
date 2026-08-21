"""ModelArchitecture 명세를 실제 신경망으로 조립한다. (실습 3-2)

Domain 은 "어떤 모양의 계산을 할 것인가"만 말했다.
그것을 nn.Module 로 바꾸는 일이 여기서 일어난다.

그리고 층별 출력 모양·파라미터 수·연산량을 센다.
이 숫자들이 모듈 4(최적화)에서 그대로 다시 쓰인다.
"""

from __future__ import annotations

import torch
from torch import nn

from domain.model.architecture import (
    ArchitectureKind,
    ArchitectureProfile,
    GlobalPooling,
    LayerProfile,
    ModelArchitecture,
)


class TimeSeriesCNN(nn.Module):
    """(L, C) 입력을 받아 클래스 점수를 낸다.

    forward 첫 줄의 transpose 가 중요하다.
    데이터는 (시간, 채널) 순서인데 Conv1d 는 (채널, 시간) 을 기대한다.
    이 한 줄을 빼먹으면 '시간'을 '채널'로 읽으면서도 학습은 그냥 돌아간다.
    """

    def __init__(self, architecture: ModelArchitecture) -> None:
        super().__init__()
        length, channels = architecture.input_spec.shape
        blocks: list[nn.Module] = []
        in_channels = channels
        for out_channels in architecture.hidden_channels:
            blocks.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=architecture.kernel_size,
                    padding=architecture.kernel_size // 2,
                )
            )
            blocks.append(nn.ReLU())
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.pool = (
            nn.AdaptiveMaxPool1d(1)
            if architecture.pooling is GlobalPooling.MAX
            else nn.AdaptiveAvgPool1d(1)
        )
        self.dropout = nn.Dropout(architecture.dropout)
        self.classifier = nn.Linear(in_channels, architecture.class_count)
        self._length = length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, L, C) → (B, C, L)
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(self.dropout(x))


class FlattenMLP(nn.Module):
    """시간 축을 펼쳐 버리는 모델.

    순서를 못 본다. 그래서 시계열에는 대개 부족하다.
    그 '부족함'을 눈으로 확인하는 것이 이 클래스의 교육적 쓸모다.
    """

    def __init__(self, architecture: ModelArchitecture) -> None:
        super().__init__()
        size = architecture.input_spec.element_count
        layers: list[nn.Module] = [nn.Flatten()]
        in_features = size
        for out_features in architecture.hidden_channels:
            layers += [nn.Linear(in_features, out_features), nn.ReLU()]
            in_features = out_features
        layers += [
            nn.Dropout(architecture.dropout),
            nn.Linear(in_features, architecture.class_count),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageCNN(nn.Module):
    """(C, H, W) 입력을 받는다. (실습 3-3)"""

    def __init__(self, architecture: ModelArchitecture) -> None:
        super().__init__()
        channels, _, _ = architecture.input_spec.shape
        blocks: list[nn.Module] = []
        in_channels = channels
        for out_channels in architecture.hidden_channels:
            blocks += [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=architecture.kernel_size,
                    padding=architecture.kernel_size // 2,
                ),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.pool = (
            nn.AdaptiveMaxPool2d(1)
            if architecture.pooling is GlobalPooling.MAX
            else nn.AdaptiveAvgPool2d(1)
        )
        self.dropout = nn.Dropout(architecture.dropout)
        self.classifier = nn.Linear(in_channels, architecture.class_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(self.dropout(x))


def build_module(architecture: ModelArchitecture) -> nn.Module:
    """명세 → 신경망. 이 함수가 Domain 과 PyTorch 의 유일한 접점이다."""
    if architecture.kind is ArchitectureKind.CNN1D:
        return TimeSeriesCNN(architecture)
    if architecture.kind is ArchitectureKind.CNN2D:
        return ImageCNN(architecture)
    return FlattenMLP(architecture)


class TorchArchitectureProfiler:
    """domain.model.ports.ArchitectureProfiler 구현. (실습 3-2)

    dummy 입력을 한 번 흘려보내며 층마다 무엇이 나오는지 기록한다.
    "AI가 데이터를 계산하는 과정을 직접 뜯어보자"가 이 클래스다.
    """

    def profile(self, architecture: ModelArchitecture) -> ArchitectureProfile:
        module = build_module(architecture)
        module.eval()

        records: list[LayerProfile] = []
        handles = []

        def make_hook(name: str, layer: nn.Module):  # noqa: ANN202
            def hook(_module, _inputs, output) -> None:  # noqa: ANN001
                shape = tuple(int(d) for d in output.shape[1:])
                records.append(
                    LayerProfile(
                        name=name,
                        kind=type(layer).__name__,
                        output_shape=shape,
                        parameter_count=sum(
                            p.numel() for p in layer.parameters(recurse=False)
                        ),
                        mac_count=_mac_count(layer, output),
                    )
                )

            return hook

        for name, layer in module.named_modules():
            if not name or list(layer.children()):
                continue  # 컨테이너는 건너뛴다. 잎(leaf) 층만 센다.
            handles.append(layer.register_forward_hook(make_hook(name, layer)))

        with torch.no_grad():
            dummy = torch.zeros(1, *architecture.input_spec.shape)
            output = module(dummy)

        for handle in handles:
            handle.remove()

        return ArchitectureProfile(
            layers=tuple(records),
            input_shape=architecture.input_spec.shape,
            output_shape=tuple(int(d) for d in output.shape[1:]),
        )


def _mac_count(layer: nn.Module, output: torch.Tensor) -> int:
    """층 하나가 필요로 하는 곱셈-덧셈 수.

    Conv 는 '출력 화소 하나당 커널 크기 × 입력 채널' 만큼 곱셈이 필요하다.
    Linear 는 '출력 하나당 입력 수' 다.
    """
    if isinstance(layer, nn.Linear):
        return int(layer.in_features * layer.out_features)
    if isinstance(layer, (nn.Conv1d, nn.Conv2d)):
        kernel = 1
        for k in layer.kernel_size:
            kernel *= k
        output_elements = 1
        for dim in output.shape[2:]:
            output_elements *= int(dim)
        return int(
            kernel * (layer.in_channels // layer.groups) * layer.out_channels * output_elements
        )
    return 0
