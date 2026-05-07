"""DetailedCNN - 带 BatchNorm 和 Dropout 的标准网络

在 SimpleCNN 基础上增加：
- 第三个卷积块（128通道）
- 每层 BatchNorm 稳定训练
- Dropout 抑制过拟合
- FC 层 BatchNorm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetailedCNN(nn.Module):
    """生产级 CNN，针对 64×64 灰度图优化"""

    def __init__(self, num_classes: int = 62, dropout_rate: float = 0.5):
        super().__init__()
        # Block 1: 1→32, 64×64
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Block 2: 32→64, 32×32
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Block 3: 64→128, 16×16
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        # 分类器：128×8×8 = 8192 → 512 → 62
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [B, 32, 32, 32]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [B, 64, 16, 16]
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # [B, 128, 8, 8]
        x = x.view(x.size(0), -1)                        # [B, 8192]
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)                                  # [B, 62]
        return x
