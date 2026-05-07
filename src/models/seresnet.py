"""SEResNet - 带 SE-Block 通道注意力的残差网络

在 ResNet 基础上集成 Squeeze-and-Excitation 注意力机制，
自动学习哪些特征通道更重要，增强对易混淆字符（0/O、1/l）的区分能力。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """通道注意力模块

    1. Squeeze: 全局平均池化压缩空间信息
    2. Excitation: 两层 FC 学习通道依赖
    3. Scale: 用 sigmoid 权重重新校准通道
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.size()
        y = self.squeeze(x).view(batch, channels)
        y = self.excitation(y)
        y = y.view(batch, channels, 1, 1)
        return x * y.expand_as(x)


class SEResidualBlock(nn.Module):
    """带 SE-Block 的残差块"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels, reduction=16) if use_se else None

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se is not None:
            out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SEResNet(nn.Module):
    """SE-ResNet，4-stage，通道 32→256，针对 64×64 灰度图"""

    def __init__(self, num_classes: int = 62):
        super().__init__()
        self.in_channels = 32

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.layer1 = self._make_layer(32, num_blocks=2, stride=1, use_se=True)
        self.layer2 = self._make_layer(64, num_blocks=2, stride=2, use_se=True)
        self.layer3 = self._make_layer(128, num_blocks=2, stride=2, use_se=True)
        self.layer4 = self._make_layer(256, num_blocks=2, stride=2, use_se=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * 1 * 1, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int, use_se: bool = True):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        in_channels = self.in_channels
        for s in strides:
            layers.append(SEResidualBlock(in_channels, out_channels, s, use_se=use_se))
            in_channels = out_channels
        self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
