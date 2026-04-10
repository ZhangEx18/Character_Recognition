"""
CNN 模型定义 - 深度架构详解 (支持 64x64 输入)

========================================================================
                    CNN 架构完整数据流 (以 SimpleCNN 为例)
========================================================================

输入图像 (1×64×64)
       ↓
┌──────────────────────────────────────────────────────────────────────┐
│  第一阶段：特征提取（卷积层 + 激活函数 + 池化层）                      │
│                                                                      │
│  卷积层：用卷积核扫描图像，提取局部特征（边缘、纹理、形状等）          │
│  激活函数：引入非线性，让网络能学习复杂模式（ReLU：负值变0，正值不变）│
│  池化层：缩小特征图尺寸，保留主要特征，减少计算量                     │
│                                                                      │
│  Conv1 → ReLU → Pool: (1,64,64) → (32,64,64) → (32,32,32)           │
│  Conv2 → ReLU → Pool: (32,32,32) → (64,32,32) → (64,16,16)          │
└──────────────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────────────┐
│  第二阶段：空间展平                                                  │
│                                                                      │
│  将多维特征图"拉直"成一维向量，以便输入全连接层                       │
│  (64,16,16) → (16384,)                                              │
└──────────────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────────────┐
│  第三阶段：分类识别（全连接层）                                      │
│                                                                      │
│  FC1：特征组合与降维，提取高级语义特征                               │
│  FC2：输出层，每个神经元对应一个类别的得分                          │
│                                                                      │
│  FC1 + ReLU: (16384,) → (512,)                                      │
│  FC2: (512,) → (62,)  ← 62个类别（0-9, a-z, A-Z）                   │
└──────────────────────────────────────────────────────────────────────┘
       ↓
输出结果：62个类别的概率分布

========================================================================
                      核心网络层详细说明
========================================================================

1. 卷积层 (Convolutional Layer)
   - 作用：用小窗口（卷积核）在图像上滑动，提取局部特征。
   - 参数说明：
     * in_channels：输入的通道数（灰度图=1，彩色图=3）。
     * out_channels：卷积核数量（决定提取多少种特征）。
     * kernel_size：卷积核大小（常见 3×3 或 5×5）。
     * stride：滑动步长（默认1）。
     * padding：边缘填充（保持输出尺寸）。

2. 池化层 (Pooling Layer)
   - 作用：降维，减少计算量，扩大感受野，提取主要特征并增加平移不变性。
   - 常见类型：最大池化（保留最显著特征）、平均池化（保留背景信息）。

3. 批量归一化层 (Batch Normalization Layer) - 见 DetailedCNN
   - 作用：强制将数据拉回均值为0、方差为1的正态分布，加速收敛，防止梯度消失。

4. 全连接层 (Fully Connected Layer)
   - 作用：将局部特征组合为全局特征，映射到类别空间完成分类。
========================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SimpleCNN(nn.Module):
    """
    简单的 CNN 网络 (针对 64x64 分辨率优化)

    网络结构：
    ┌─────────────────────────────────────────────────────────────────┐
    │  输入: (batch_size, 1, 64, 64)  # 1通道 64×64 灰度图            │
    ├─────────────────────────────────────────────────────────────────┤
    │  特征提取阶段:                                                  │
    │  Conv1(1→32, 3×3) → ReLU → MaxPool(2×2)                       │
    │  Conv2(32→64, 3×3) → ReLU → MaxPool(2×2)                      │
    ├─────────────────────────────────────────────────────────────────┤
    │  空间展平:                                                      │
    │  Flatten: (64, 16, 16) → (16384,)                              │
    ├─────────────────────────────────────────────────────────────────┤
    │  分类识别阶段:                                                  │
    │  FC1(16384→512) → ReLU                                         │
    │  FC2(512→62)  # 62个类别                                       │
    ├─────────────────────────────────────────────────────────────────┤
    │  输出: (batch_size, 62)  # 每个类别的得分                       │
    └─────────────────────────────────────────────────────────────────┘
    """

    NUM_CLASSES = 62  # 总类别数：数字(10) + 小写字母(26) + 大写字母(26)
    NUM_DIGITS = 10   # 数字类别数 (0-9)
    NUM_LETTERS = 26  # 字母类别数 (a-z, A-Z)

    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        # ----- 第一个卷积层 -----
        # 输入: 1个通道(灰度图)。输出: 32个特征图。
        # 尺寸变化: 64x64输入，padding=1保持尺寸，输出仍为 64x64
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)

        # ----- 第二个卷积层 -----
        # 接收 conv1 经过池化后的特征图(32通道)。输出: 64个高级特征图。
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # ----- 池化层 -----
        # 2x2最大池化，每次经过都会让特征图的长宽减半
        self.pool = nn.MaxPool2d(2, 2)

        # ----- 全连接层 -----
        # 输入展平: 经过两次池化，64x64 -> 32x32 -> 16x16
        # 通道数为64，因此总维度为 64 * 16 * 16 = 16384
        self.fc1 = nn.Linear(64 * 16 * 16, 512)  # 第一层将 16384 维压缩至 512 维
        self.fc2 = nn.Linear(512, num_classes)   # 输出层映射到具体类别数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播过程"""
        # 第一阶段：特征提取
        # [Batch, 1, 64, 64] -> Conv1 -> [Batch, 32, 64, 64]
        # -> ReLU -> Pool -> [Batch, 32, 32, 32]
        x = self.pool(F.relu(self.conv1(x)))

        # [Batch, 32, 32, 32] -> Conv2 -> [Batch, 64, 32, 32]
        # -> ReLU -> Pool -> [Batch, 64, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))

        # 第二阶段：空间展平
        # 将三维特征图 [Batch, 64, 16, 16] 展开为一维向量 [Batch, 16384]
        x = x.view(x.size(0), -1)

        # 第三阶段：分类识别
        # [Batch, 16384] -> FC1 -> ReLU -> [Batch, 512]
        x = F.relu(self.fc1(x))

        # [Batch, 512] -> FC2 -> [Batch, 62] (输出原始得分 logits)
        x = self.fc2(x)

        return x

    @staticmethod
    def get_class_name(class_idx: int) -> str:
        """根据类别索引获取类别名称 (0-9, a-z, A-Z)"""
        if 0 <= class_idx < 10:
            return str(class_idx)
        elif 10 <= class_idx < 36:
            return chr(ord('a') + class_idx - 10)
        elif 36 <= class_idx < 62:
            return chr(ord('A') + class_idx - 36)
        else:
            return f"Unknown({class_idx})"


class DetailedCNN(nn.Module):
    """
    加强版 CNN 网络 (支持更深层次特征提取，加入正则化机制)
    相比 SimpleCNN 增加了：第三层卷积、批量归一化(BatchNorm)、Dropout
    """
    def __init__(self, num_classes: int = 62, dropout_rate: float = 0.5):
        super(DetailedCNN, self).__init__()

        # ----- 第一层卷积块 -----
        # 尺寸: 64x64 -> 池化后 -> 32x32
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 标准化卷积输出，加速训练，使模型对超参数不那么敏感

        # ----- 第二层卷积块 -----
        # 尺寸: 32x32 -> 池化后 -> 16x16
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # ----- 第三层卷积块 (新增) -----
        # 提取更深层次的抽象特征
        # 尺寸: 16x16 -> 池化后 -> 8x8
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        # ----- 全连接层 -----
        # 经过3次池化：64 -> 32 -> 16 -> 8
        # 展平维度计算：128个通道 * 8 * 8 = 8192
        self.fc1 = nn.Linear(128 * 8 * 8, 512)

        # 针对全连接层的一维 BatchNorm
        # 核心作用：防止数据偏移进入 ReLU 的负区间导致神经元失活（即"ReLU 死亡"）
        self.bn_fc1 = nn.BatchNorm1d(512)

        # Dropout 层：在训练时随机按比例（默认50%）关闭部分神经元
        # 作用：强制网络学习更加鲁棒的特征，防止过拟合
        self.dropout = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播过程：严格按照 Conv -> BN -> ReLU -> Pool 顺序"""
        # 第一层: 输入 64x64 -> 输出特征图 32x32
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # 第二层: 输入 32x32 -> 输出特征图 16x16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # 第三层: 输入 16x16 -> 输出特征图 8x8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # 自适应展平: 将多维特征拉平成一维 [Batch, 8192]
        x = x.view(x.size(0), -1)

        # 全连接层的执行顺序：Linear -> BatchNorm -> ReLU -> Dropout
        x = self.fc1(x)
        x = self.bn_fc1(x)  # 强制拉回正态分布，确保均值为0，方差为1
        x = F.relu(x)       # 此时刚好有约 50% 的值大于 0，被 ReLU 完美激活
        x = self.dropout(x) # 训练阶段随机断开连接，测试阶段自动失效

        x = self.fc2(x)     # 输出分类 logits: [Batch, 62]
        return x


def count_parameters(model: nn.Module) -> int:
    """计算模型中需要梯度更新的可训练参数总数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_output_shape(model: nn.Module, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """计算模型输出形状（用于验证网络架构尺寸的正确性）"""
    model.eval()
    with torch.no_grad():
        # 添加 batch 维度 (batch_size=1) 模拟实际输入
        dummy_input = torch.zeros(1, *input_shape)
        output = model(dummy_input)
        return output.shape[1:]  # 移除 batch 维度返回


if __name__ == "__main__":
    # ==================== 模型测试 ====================
    print("=" * 60)
    print("CNN 三层架构模型测试")
    print("=" * 60)

    # 创建 SimpleCNN 模型
    model = SimpleCNN(num_classes=62)
    print("\n[SimpleCNN] 网络结构：")
    print(model)
    print(f"\n[SimpleCNN] 总参数量: {count_parameters(model):,}")

    # 创建 DetailedCNN 模型进行对比
    detailed_model = DetailedCNN(num_classes=62)
    print(f"\n[DetailedCNN] 总参数量: {count_parameters(detailed_model):,}")

    # 测试前向传播 (修正了 dummy_input 的尺寸为 64x64 以匹配模型定义)
    print("\n测试前向传播 (以 SimpleCNN 为例)：")
    dummy_input = torch.randn(4, 1, 64, 64)  # batch_size=4, channels=1, 64x64 图片
    output = model(dummy_input)
    print(f"  输入形状: {dummy_input.shape}")
    print(f"  输出形状: {output.shape} (预期: [4, 62])")

    # 测试类别索引转名称
    print("\n类别索引解析示例：")
    for idx in [0, 5, 9, 10, 35, 36, 61]:
        print(f"  索引 {idx:2d} -> 类别标签: '{SimpleCNN.get_class_name(idx)}'")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)