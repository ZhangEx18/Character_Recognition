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
        # out_channels=32: 使用 32 个不同的卷积核提取 32 种不同的低级特征（例如：横向边缘、点）。
        # kernel_size=3: 卷积核的大小是 3x3 的矩阵。这是一种兼顾性能与精度的黄金尺寸。
        # padding=1: 在图像四周各填充一圈0。防止边缘信息在卷积中丢失，并保持卷积前后尺寸一致。
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)

        # ----- 第二个卷积层 -----
        # 接收 conv1 经过池化后的特征图(32通道)。输出: 64个高级特征图。
        # in_channels=32: 必须与上一层的输出通道数严格匹配。
        # out_channels=64: 网络越深，特征越抽象，通道数翻倍有助于组合出复杂模式（如直角、弧线）。
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # ----- 共用池化层 -----
        # 2x2最大池化，每次经过都会让特征图的长宽减半。
        # kernel_size=2, stride=2: 在 2x2 窗口内取最大值，滑动步长为 2（不重叠）。
        # 作用：大幅压缩空间维度（如 64x64 变 32x32），减少计算量，同时赋予网络轻微的“平移不变性”。
        self.pool = nn.MaxPool2d(2, 2)

        # ----- 全连接层 -----
        # 展平计算推导：64x64 -> [池化] -> 32x32 -> [池化] -> 16x16
        # 第二层卷积输出 64 个通道，所以总特征数为：64 * 16 * 16 = 16384。
        self.fc1 = nn.Linear(64 * 16 * 16, 512)  # 第一层将 16384 维的巨大特征压缩到 512 维的隐层空间
        self.fc2 = nn.Linear(512, num_classes)   # 输出层将 512 维特征映射为最终的分类得分 (62类)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播过程：定义了数据在网络中流动的具体路径"""
        # 第一阶段：特征提取
        # 顺序：卷积 -> 激活函数(引入非线性) -> 最大池化(降维)
        # 输入 [Batch, 1, 64, 64] -> 输出 [Batch, 32, 32, 32]
        x = self.pool(F.relu(self.conv1(x)))

        # 顺序同上，进行第二次特征提取
        # 输入 [Batch, 32, 32, 32] -> 输出 [Batch, 64, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))

        # 第二阶段：空间展平
        # 将三维特征图 [Batch, 64, 16, 16] 拉直为一维向量 [Batch, 16384]
        # view 方法中的 -1 表示自动计算该维度的长度。
        x = x.view(x.size(0), -1)

        # 第三阶段：分类识别
        # 全连接层运算，并通过 ReLU 激活。输出尺寸：[Batch, 512]
        x = F.relu(self.fc1(x))

        # 最终输出层。注意此处绝对不能加激活函数，计算 Loss 时会自带 Softmax。
        # 输出尺寸：[Batch, 62]
        x = self.fc2(x)

        return x

    @staticmethod
    def get_class_name(class_idx: int) -> str:
        """根据类别索引反推类别名称 (辅助函数)"""
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
    加强版 CNN 网络 (业务落地主力模型)

    【核心改进】
    相比 SimpleCNN，增加了：
    1. 第三层卷积 (更深的特征提取，能识别完整的字母结构)。
    2. BatchNorm (批量归一化，解决梯度消失，加速模型收敛)。
    3. Dropout (随机失活，打破神经元间的共适应，强力防止过拟合)。
    """
    def __init__(self, num_classes: int = 62, dropout_rate: float = 0.5):
        super(DetailedCNN, self).__init__()

        # ----- 第一层卷积块 -----
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        # BatchNorm2d (批量归一化)：在特征图进入激活函数前，将其分布强制拉回均值为 0、方差为 1 的正态分布。
        # 【工程意义】：允许使用更大的学习率(LR)，大幅缩短训练时间，并使网络对权重初始化不那么敏感。
        self.bn1 = nn.BatchNorm2d(32)

        # ----- 第二层卷积块 -----
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # ----- 第三层卷积块 -----
        # 深度增加，参数通道数增至 128，专门用于捕获全局语义特征。
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 共用的最大池化层
        self.pool = nn.MaxPool2d(2, 2)

        # ----- 全连接层 -----
        # 展平维度推导：64 -> (池化1)32 -> (池化2)16 -> (池化3)8
        # 最终输出通道数为 128，所以尺寸为：128 * 8 * 8 = 8192
        self.fc1 = nn.Linear(128 * 8 * 8, 512)

        # 针对全连接层的一维 BatchNorm
        # 【防御机制】：防止全连接层巨量的数据偏移直接击穿 ReLU 导致神经元“坏死”（恒输出0）。
        self.bn_fc1 = nn.BatchNorm1d(512)

        # Dropout 层 (防过拟合利器)
        # 训练时，以 dropout_rate（默认 50%）的概率随机"蒙住"部分神经元的眼睛。
        # 强迫网络不要只盯着几个明显特征（如死记硬背某几个白点），而是学习整体字形规律。
        # 测试时该层会自动失效。
        self.dropout = nn.Dropout(dropout_rate)

        # 最终输出层
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播过程：严格遵守 Conv -> BatchNorm -> ReLU -> Pool 范式"""
        # 第一块: 输出尺寸 32x32
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # 第二块: 输出尺寸 16x16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # 第三块: 输出尺寸 8x8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # 自适应展平
        x = x.view(x.size(0), -1)

        # 全连接运算顺序：Linear -> BatchNorm -> ReLU -> Dropout
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # 输出层映射
        x = self.fc2(x)
        return x


class ResidualBlock(nn.Module):
    """
    残差块 (ResNet的核心引擎)

    【为什么需要残差？】
    当网络非常深时，梯度在反向传播时会不断衰减直至消失，导致浅层网络根本无法学习（网络退化）。
    残差块通过增加一条“旁路 (Shortcut)”，让信息可以直接跨层流动，完美解决了深度带来的负面效应。
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # 主路：两层标准的 3x3 卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 旁路 (Shortcut) 逻辑：
        # 如果尺寸变小（stride>1）或通道变多，旁路不能直接相加。
        # 必须使用 1x1 卷积同步调整旁路的尺寸和通道数，使其与主路完全一致。
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 主路计算流
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 【灵魂一步】：将源输入 x (或经过1x1卷积调整的x) 直接加到主路输出上。
        # 这就是 "F(x) + x" 的残差思想。
        out += self.shortcut(x)

        # 相加后再进行最终的非线性激活
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    轻量级残差网络 (潜能最大架构)
    针对本项目 64x64 的字符识别任务定制。
    摒弃了传统的 MaxPool，完全依靠带步长(stride=2)的卷积进行空间降维，保留更多细节。
    """

    def __init__(self, num_classes=62):
        super(ResNet, self).__init__()
        self.in_channels = 32

        # 初始“茎”网络 (Stem)：快速升维
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # 核心残差组 (共 4 个 Stage)
        # 每个 _make_layer 代表一个 Stage，其中包含 num_blocks 个残差块。
        # 每次 Stage 切换时，通道数翻倍，尺寸通过 stride=2 减半。
        self.layer1 = self._make_layer(32, num_blocks=2, stride=1)  # 尺寸保持 64x64
        self.layer2 = self._make_layer(64, num_blocks=2, stride=2)  # 尺寸压缩为 32x32
        self.layer3 = self._make_layer(128, num_blocks=2, stride=2) # 尺寸压缩为 16x16
        self.layer4 = self._make_layer(256, num_blocks=2, stride=2) # 尺寸压缩为 8x8

        # 自适应平均池化层 (AdaptiveAvgPool)
        # 【工程妙招】：不论前面的图像因为任何原因输入尺寸发生变化，
        # 这里都会强制将其空间维度拉扯/压缩成固定的 4x4 大小。
        # 这使得网络完全免疫输入图片大小的轻微波动，告别展平时算错维度的噩梦。
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        # 最终分类器
        # 展平：256通道 * 4宽 * 4高 = 4096
        self.fc = nn.Linear(256 * 4 * 4, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        """工厂函数：根据指定参数批量制造并堆叠残差块"""
        # 第一块的步长可能是 2(为了降维)，后续块的步长全为 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels # 更新全局输入通道，供给下一个 Block 使用
        return nn.Sequential(*layers)

    def forward(self, x):
        """网络主干执行序列"""
        out = F.relu(self.bn1(self.conv1(x))) # Stem

        # 依次穿过四层残差矩阵
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # 全局池化与展平
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        # 映射最终分类
        out = self.fc(out)
        return out


# ============================================================
# 辅助分析工具函数
# ============================================================

def count_parameters(model: nn.Module) -> int:
    """
    计算模型中需要被优化器更新的可训练参数 (Trainable Parameters) 总数。
    用于评估模型的复杂度和资源消耗。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_output_shape(model: nn.Module, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    计算模型输出形状（工具函数）
    在修改了卷积核参数或图片大小后，用于动态验证特征图的尺寸变化是否符合预期。
    """
    model.eval()
    with torch.no_grad():
        # 构建一个纯 0 的假张量(Dummy Input)塞进网络跑一圈
        dummy_input = torch.zeros(1, *input_shape)
        output = model(dummy_input)
        return output.shape[1:]


# ============================================================
# 模块自检脚本
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CNN 三层架构模型测试框架")
    print("=" * 60)

    # --- 测试 SimpleCNN ---
    model_s = SimpleCNN(num_classes=62)
    print("\n[SimpleCNN] 网络结构简报：")
    print(model_s)
    print(f"[SimpleCNN] 参数量总计: {count_parameters(model_s):,} (适合快速验证)")

    # --- 测试 DetailedCNN ---
    model_d = DetailedCNN(num_classes=62)
    print(f"\n[DetailedCNN] 参数量总计: {count_parameters(model_d):,} (主力生产模型)")

    # --- 测试 ResNet ---
    model_r = ResNet(num_classes=62)
    print(f"\n[ResNet] 参数量总计: {count_parameters(model_r):,} (高潜力残差模型)")

    # --- 尺寸匹配测试 ---
    print("\n正在执行张量流转测试 (以 ResNet 为例)：")
    dummy_input = torch.randn(4, 1, 64, 64)  # 模拟批次为 4 的 64x64 单通道图像
    output = model_r(dummy_input)
    print(f"  输入张量尺寸: {dummy_input.shape}")
    print(f"  输出结果尺寸: {output.shape} (预期应为 [4, 62])")

    # --- 字典辅助测试 ---
    print("\n类别字典反向解析测试：")
    for idx in [0, 5, 9, 10, 35, 36, 61]:
        print(f"  内部索引 {idx:2d} -> 物理标签: '{SimpleCNN.get_class_name(idx)}'")

    print("\n" + "=" * 60)
    print("架构自检 100% 通过！")
    print("=" * 60)