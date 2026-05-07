"""SimpleCNN - 浅层基准网络

结构：Conv1(32) → Pool → Conv2(64) → Pool → Flatten → FC1(512) → FC2(62)
参数量最小，适合快速验证和基准对比。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """基础 CNN，针对 64×64 单通道灰度图优化"""

    NUM_CLASSES = 62
    NUM_DIGITS = 10
    NUM_LETTERS = 26

    def __init__(self, num_classes: int = 62):
        super().__init__()
        # 浅层特征：1→32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 深层特征：32→64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 下采样
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 分类器：64×16×16 = 16384 → 512 → 62
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # [B, 32, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))   # [B, 64, 16, 16]
        x = x.view(x.size(0), -1)              # [B, 16384]
        x = F.relu(self.fc1(x))                # [B, 512]
        x = self.fc2(x)                        # [B, 62]
        return x

    @staticmethod
    def get_class_name(class_idx: int) -> str:
        """索引转字符"""
        if 0 <= class_idx < 10:
            return str(class_idx)
        elif 10 <= class_idx < 36:
            return chr(ord('a') + class_idx - 10)
        elif 36 <= class_idx < 62:
            return chr(ord('A') + class_idx - 36)
        else:
            return f"Unknown({class_idx})"
