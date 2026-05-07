"""MobileNet - 深度可分离卷积轻量网络

2017 年 Google 提出，核心创新：Depthwise Separable Convolution
将标准卷积分解为 Depthwise（逐通道卷积）+ Pointwise（1×1 卷积），
大幅减少参数量和计算量，适合移动端和边缘设备。

结构：Conv → DW+PW×多个 stage → GAP → FC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积：Depthwise + Pointwise"""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        # Depthwise: 逐通道卷积，groups=in_ch
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3,
                                   stride=stride, padding=1, groups=in_ch,
                                   bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        # Pointwise: 1×1 卷积，跨通道融合
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1,
                                   stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x


class MobileNet(nn.Module):
    """MobileNet 适配版，64×64 灰度图，62 类"""

    def __init__(self, num_classes: int = 62):
        super().__init__()
        # 标准卷积开头
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2,
                               padding=1, bias=False)  # 64→32
        self.bn1 = nn.BatchNorm2d(32)

        # Depthwise Separable 层组
        self.layers = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=1),    # 32×32
            DepthwiseSeparableConv(64, 128, stride=2),   # 32→16
            DepthwiseSeparableConv(128, 128, stride=1),  # 16×16
            DepthwiseSeparableConv(128, 256, stride=2),  # 16→8
            DepthwiseSeparableConv(256, 256, stride=1),  # 8×8
            DepthwiseSeparableConv(256, 512, stride=2),  # 8→4
            DepthwiseSeparableConv(512, 512, stride=1),  # 4×4
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
