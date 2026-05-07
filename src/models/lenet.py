"""LeNet-5 - 卷积神经网络始祖

1998 年 Yann LeCun 提出，手写数字识别的开创性工作。
原始结构针对 32×32 灰度图，此处适配到 64×64 和 62 类输出。

结构：Conv(1→6)→Pool → Conv(6→16)→Pool → FC(→120)→FC(→84)→FC(→62)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """LeNet-5 适配版，64×64 灰度图，62 类输出"""

    def __init__(self, num_classes: int = 62):
        super().__init__()
        # 特征提取：两层卷积 + 平均池化
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)   # 64×64
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0)  # 32×32 → 28×28
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # 分类器：三层全连接
        # 经过两次 pool: 64→32→14, 16通道, 14×14=196, 16×196=3136
        self.fc1 = nn.Linear(16 * 14 * 14, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # [B,6,64,64] → [B,6,32,32]
        x = self.pool(F.relu(self.conv2(x)))   # [B,16,28,28] → [B,16,14,14]
        x = x.view(x.size(0), -1)              # [B,16×14×14=3136]
        x = F.relu(self.fc1(x))                # [B,120]
        x = F.relu(self.fc2(x))                # [B,84]
        x = self.fc3(x)                        # [B,62]
        return x
