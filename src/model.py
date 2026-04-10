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
        # in_channels=1: 因为输入是灰度图，只有一个颜色通道。
        # out_channels=32: 我们使用 32 个不同的卷积核（过滤器）来提取 32 种不同的特征（例如：横向边缘、纵向边缘等）。
        # kernel_size=3: 卷积核的大小是 3x3 的矩阵。这是一种非常常见且有效的尺寸。
        # padding=1: 在图像四周各填充一圈像素（通常是0）。因为 3x3 卷积核在扫描时会使图像尺寸缩小 2，
        # 为了保证卷积操作前后图像尺寸不变（仍为 64x64），需要加 padding。公式：(64 - 3 + 2*1)/1 + 1 = 64
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)

        # ----- 第二个卷积层 -----
        # 接收 conv1 经过池化后的特征图(32通道)。输出: 64个高级特征图。
        # in_channels=32: 必须与上一层的输出通道数匹配。
        # out_channels=64: 网络越深，我们希望提取的特征越抽象、数量越多。
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # ----- 池化层 -----
        # 2x2最大池化，每次经过都会让特征图的长宽减半
        # kernel_size=2, stride=2: 表示在 2x2 的窗口内取最大值，并且每次滑动 2 个像素（不重叠）。
        # 这会将 64x64 的图像压缩成 32x32，大大减少计算量，同时保留最显著的特征（如边缘）。
        self.pool = nn.MaxPool2d(2, 2)

        # ----- 全连接层 -----
        # 输入展平: 经过两次池化，长宽变化为：64x64 -> (一次池化) -> 32x32 -> (二次池化) -> 16x16
        # 因为在进入全连接层之前，我们需要把三维的特征图（通道数 x 高 x 宽）拉平成一维向量。
        # 通道数为64，因此总维度（特征数量）为 64 * 16 * 16 = 16384。
        # 这个数字必须算得非常精准，否则会报错。
        self.fc1 = nn.Linear(64 * 16 * 16, 512)  # 第一层将 16384 维的特征压缩/映射到 512 维的隐层空间
        self.fc2 = nn.Linear(512, num_classes)   # 输出层将 512 维的特征映射到最终的类别数（如 62 类），每个节点代表该类的得分

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播过程：定义了数据在网络中流动的具体路径"""
        # 第一阶段：特征提取
        # 步骤 1: self.conv1(x) -> 进行第一次卷积，尺寸变为 [Batch, 32, 64, 64]
        # 步骤 2: F.relu(...) -> 应用 ReLU 激活函数，将所有负数变为 0，引入非线性
        # 步骤 3: self.pool(...) -> 进行最大池化，尺寸长宽减半，变为 [Batch, 32, 32, 32]
        x = self.pool(F.relu(self.conv1(x)))

        # 同理，进行第二次特征提取
        # 经过 conv2，通道数变为 64；经过 pool，尺寸再次减半。
        # 最终输出尺寸：[Batch, 64, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))

        # 第二阶段：空间展平
        # 将三维特征图 [Batch, 64, 16, 16] 展开为一维向量 [Batch, 16384]
        # x.size(0) 是 batch_size（比如 128）。
        # -1 告诉 PyTorch："除了 batch 维度，把剩下的所有维度乘起来拼成一维"。
        x = x.view(x.size(0), -1)

        # 第三阶段：分类识别
        # 将 16384 维的特征输入到第一个全连接层 fc1，并通过 ReLU 激活。
        # 输出尺寸：[Batch, 512]
        x = F.relu(self.fc1(x))

        # 将 512 维的特征输入到输出层 fc2，得到最终每个类别的原始得分（logits）。
        # 注意：这里一般不加激活函数，因为在计算损失（如 CrossEntropyLoss）时会自动应用 Softmax。
        # 输出尺寸：[Batch, 62]
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
    相比 SimpleCNN 增加了：第三层卷积、批量归一化(BatchNorm)、Dropout。
    这是本项目实际训练和推断使用的主力模型。
    """
    def __init__(self, num_classes: int = 62, dropout_rate: float = 0.5):
        super(DetailedCNN, self).__init__()

        # ----- 第一层卷积块 -----
        # 输入 1 通道，输出 32 通道。
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        # BatchNorm2d (批量归一化)：在特征图进入激活函数前，将其分布强制拉回均值为 0、方差为 1 的正态分布。
        # 作用极其重要：1. 解决梯度消失问题；2. 允许使用更大的学习率，大幅加速训练；3. 降低模型对初始权重的敏感度。
        self.bn1 = nn.BatchNorm2d(32)

        # ----- 第二层卷积块 -----
        # 输入 32 通道，输出 64 通道。
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # ----- 第三层卷积块 (新增) -----
        # 相比 SimpleCNN，增加了一层卷积，旨在提取更深层次、更高级的抽象特征（例如完整字母的形状）。
        # 输入 64 通道，输出 128 通道。
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 共用的最大池化层
        self.pool = nn.MaxPool2d(2, 2)

        # ----- 全连接层 -----
        # 展平维度计算：
        # 图像初始大小为 64x64。经过 3 次 2x2 的池化后：
        # 64 / 2 = 32
        # 32 / 2 = 16
        # 16 / 2 = 8
        # 最终特征图大小为 8x8，通道数为 128。所以展平后的总维度是 128 * 8 * 8 = 8192。
        self.fc1 = nn.Linear(128 * 8 * 8, 512)

        # 针对全连接层的一维 BatchNorm
        # 核心作用：防止全连接层输出的数据偏移进入 ReLU 的负区间，导致神经元失活（即"ReLU 死亡"现象，永远输出 0）。
        self.bn_fc1 = nn.BatchNorm1d(512)

        # Dropout 层 (随机失活)
        # 训练时，以 dropout_rate（默认 50%）的概率随机将部分神经元的输出置为 0。
        # 作用：强迫网络不要过度依赖某几个特定的特征节点，而是学习更广泛、更鲁棒的特征，是防止过拟合（死记硬背训练集）的利器。
        # 注意：在测试/推断模式下（model.eval()），Dropout 会自动失效，所有神经元都会参与计算。
        self.dropout = nn.Dropout(dropout_rate)

        # 最终输出层：从 512 维映射到具体的分类数（62类）
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播过程：严格按照 Conv -> BN -> ReLU -> Pool 顺序"""
        # 第一块: 卷积 -> 归一化 -> 激活 -> 池化
        # 输入尺寸: 64x64 -> 池化后输出特征图尺寸: 32x32
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # 第二块:
        # 输入尺寸: 32x32 -> 池化后输出特征图尺寸: 16x16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # 第三块:
        # 输入尺寸: 16x16 -> 池化后输出特征图尺寸: 8x8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # 自适应展平: 将多维特征 [Batch, 128, 8, 8] 拉平成一维 [Batch, 8192]
        x = x.view(x.size(0), -1)

        # 全连接层的执行顺序：Linear -> BatchNorm -> ReLU -> Dropout
        # 1. 线性变换计算特征组合
        x = self.fc1(x)
        # 2. 归一化，确保数据分布健康
        x = self.bn_fc1(x)
        # 3. 非线性激活
        x = F.relu(x)
        # 4. 随机丢弃部分信息，增加鲁棒性
        x = self.dropout(x)

        # 输出层，得到最终分类得分: [Batch, 62]
        x = self.fc2(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """计算模型中需要梯度更新的可训练参数总数"""
    # 遍历模型的所有参数，如果 requires_grad 为 True（即参与训练更新），则累加其元素个数（numel）
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_output_shape(model: nn.Module, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """计算模型输出形状（用于验证网络架构尺寸的正确性）"""
    # 设为评估模式，防止 Dropout 干扰
    model.eval()
    with torch.no_grad(): # 不计算梯度，节省内存
        # 添加 batch 维度 (batch_size=1) 模拟实际输入，生成一个全 0 的假张量
        dummy_input = torch.zeros(1, *input_shape)
        # 前向传播跑一次
        output = model(dummy_input)
        # 移除 batch 维度并返回形状
        return output.shape[1:]


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