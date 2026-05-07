"""MLPNet - 全连接多层感知机

源自 MNIST 手写数字识别项目的全连接网络风格，
适配到 64×64 灰度图（4096维输入）和 62 类字符识别输出。

结构：Flatten → FC(4096→512) → BN → ReLU → Drop → FC(512→256) → BN → ReLU → Drop → FC(256→62)
无卷积层，纯全连接，适合作为卷积网络的 baseline 对比。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPNet(nn.Module):
    """全连接网络，无卷积层"""

    def __init__(self, num_classes: int = 62):
        super().__init__()
        # 64×64 = 4096 维输入
        self.fc1 = nn.Linear(64 * 64, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 展平：[B, 1, 64, 64] → [B, 4096]
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
