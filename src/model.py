"""
CNN 模型定义 - 深度架构解析 (支持 64x64 空间分辨率)

========================================================================
【全景处理流程】(以 SimpleCNN 架构为例)
[起点] 输入张量 (Input Tensor): (Batch_Size, 1, 64, 64) 单通道灰度图像矩阵
  │
  ├─ 1. 浅层特征提取 (Conv1 Block)
  │   ├─ 线性映射: 3×3 卷积核提取局部基础边缘与纹理特征 (1 → 32 通道)
  │   ├─ 非线性激活: ReLU 函数激活，引入非线性表征能力
  │   └─ 空间下采样: 2×2 最大池化，压缩空间分辨率并提供局部平移不变性 (64×64 → 32×32)
  │
  ├─ 2. 深层特征提取 (Conv2 Block)
  │   ├─ 线性映射: 3×3 卷积核组合浅层特征，提取高维抽象语义 (32 → 64 通道)
  │   ├─ 非线性激活: ReLU 函数激活
  │   └─ 空间下采样: 2×2 最大池化，二次压缩空间分辨率 (32×32 → 16×16)
  │
  ├─ 3. 张量形态重构 (Flatten)
  │   └─ 维度展平: 将 (64, 16, 16) 的三维特征矩阵重构为长度 16384 的一维连续内存向量
  │
  ├─ 4. 全局语义聚合 (分类器模块)
  │   ├─ 隐层映射: 线性层 FC1 (16384 → 512) 聚合全局特征并融合 ReLU 激活
  │   └─ 类别投射: 线性层 FC2 (512 → 62) 输出各类别的对数几率 (Logits)
  │
[终点] 输出预测分布 (Output Logits): (Batch_Size, 62)

========================================================================
【核心网络层工程学说明】
1. 卷积层 (Convolutional Layer)
   - 机制：利用离散卷积运算，通过滑动局部感受野提取空间结构特征，实现权值共享以降低参数规模。
2. 池化层 (Pooling Layer)
   - 机制：执行空间维度的下采样，过滤冗余背景信息，在降低特征维度的同时增强模型对微小平移的鲁棒性。
3. 批量归一化层 (Batch Normalization Layer)
   - 机制：在批次维度对特征分布进行标准化处理，缓解内部协变量偏移 (Internal Covariate Shift)，允许使用更大的学习率并加速收敛。
4. 全连接层 (Fully Connected Layer)
   - 机制：破坏特征的空间拓扑结构，将分布式特征表示映射至特定的标签空间以完成分类。
========================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ============================================================
#  Focal Loss (焦点损失函数)
# ============================================================
class FocalLoss(nn.Module):
    """
    焦点损失函数 - 专门解决难例挖掘问题

    【核心原理】：
    标准交叉熵对所有样本一视同仁，而 Focal Loss 引入调制因子 (1 - pt)^gamma：
    - 简单样本 (pt 接近 1) 被大幅衰减权重
    - 难例样本 (pt 接近 0) 保持较高权重

    这使得模型在训练时自动把注意力集中在容易认错的"刺头"字符上，
    特别适合解决 0/O、1/l 这类易混淆字符的分类问题。

    参数:
        alpha: 正负样本平衡因子 (默认 0.25)
        gamma: 聚焦因子 (默认 2.0)，值越大越专注难例
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 计算标准交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # 获取目标类别的概率
        pt = torch.exp(-ce_loss)
        # 计算调制因子
        focal_term = (1 - pt) ** self.gamma
        # 应用 alpha 平衡和调制因子
        focal_loss = self.alpha * focal_term * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================
#  SE-Block (Squeeze-and-Excitation 注意力机制)
# ============================================================
class SEBlock(nn.Module):
    """
    通道注意力模块 (Squeeze-and-Excitation Block)

    【核心原理】：
    人类看字时，注意力是有重点的（比如区分 Q 和 O，人眼会死死盯住右下角那个小尾巴）。
    SE-Block 让网络自动学习"哪些通道的特征最重要"，给重要的通道特征加高权重，
    把没用的背景噪声权重降到极低。

    【工作机制】：
    1. Squeeze (压缩)：全局平均池化，将空间信息压缩为一个通道描述值
    2. Excitation (激发)：两层全连接网络学习通道间的非线性依赖
    3. Scale (缩放)：用学到的权重对原始特征图进行通道维度的重新校准
    """
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        # 压缩：全局平均池化，将 H×W 压缩为 1×1
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        # 激发：第一层降维 + ReLU，第二层升维 + Sigmoid
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.size()
        # Squeeze: [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        y = self.squeeze(x).view(batch, channels)
        # Excitation: [B, C] -> [B, C]
        y = self.excitation(y)
        # Scale: 重新加权到原始特征图
        y = y.view(batch, channels, 1, 1)
        return x * y.expand_as(x)


class SEResNet(nn.Module):
    """
    集成 SE-Block 的 ResNet 变体

    在每个残差块的卷积层之后插入 SE-Block，让网络学会自动判断哪些特征通道更重要，
    特别适合字符识别这种需要区分细微差异的任务（如 0 vs O，1 vs l）。
    """
    def __init__(self, num_classes: int = 62):
        super(SEResNet, self).__init__()
        self.in_channels = 32

        # Stem
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # SE-ResNet 层组
        self.layer1 = self._make_layer(32, num_blocks=2, stride=1, use_se=True)
        self.layer2 = self._make_layer(64, num_blocks=2, stride=2, use_se=True)
        self.layer3 = self._make_layer(128, num_blocks=2, stride=2, use_se=True)
        self.layer4 = self._make_layer(256, num_blocks=2, stride=2, use_se=True)

        # 全局池化与分类
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(256 * 4 * 4, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int, use_se: bool = True):
        """构建带有可选 SE-Block 的残差层"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        in_channels = self.in_channels
        for s in strides:
            layers.append(SEResidualBlock(in_channels, out_channels, s, use_se=use_se))
            in_channels = out_channels
        # 更新实例变量，确保下一层能正确接收当前层的输出通道数
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


class SEResidualBlock(nn.Module):
    """
    集成 SE-Block 的残差块

    在主路卷积之后、恒等映射相加之前插入 SE-Block 进行通道注意力校准。
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_se: bool = True):
        super(SEResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # SE-Block：通道注意力校准
        self.se = SEBlock(out_channels, reduction=16) if use_se else None

        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 应用 SE-Block 进行通道权重校准
        if self.se is not None:
            out = self.se(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SimpleCNN(nn.Module):
    """
    基础卷积神经网络架构 (针对 64x64 分辨率优化)

    【架构拓扑】：
    Input(1×64×64) -> Conv1(32) -> Pool -> Conv2(64) -> Pool -> Flatten -> FC1(512) -> FC2(62)
    """

    NUM_CLASSES = 62  # 总类别数：数字(10) + 小写字母(26) + 大写字母(26)
    NUM_DIGITS = 10   # 数字类别子集 (0-9)
    NUM_LETTERS = 26  # 字母类别子集 (a-z, A-Z)

    def __init__(self, num_classes: int):
        super(SimpleCNN, self).__init__()

        # ----- 浅层特征提取器 -----
        # 针对单通道灰度图输入，输出 32 维特征通道。
        # kernel_size=3, padding=1 的配置在保持空间维度的前提下，提供了标准的 3×3 感受野。
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)

        # ----- 深层特征提取器 -----
        # 接收 32 维特征输入，映射至 64 维特征通道，用于组合高阶语义特征。
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # ----- 空间下采样算子 -----
        # 2x2 最大池化。每次操作将输入特征图的空间分辨率长宽缩减 50%。
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ----- 全局线性分类器 -----
        # 维度推演：64×64 经过两次 2×2 池化后，空间维度降至 16×16。
        # 结合 64 个特征通道，展平后的特征向量长度为：64 × 16 × 16 = 16384。
        self.fc1 = nn.Linear(64 * 16 * 16, 512)  # 隐层维度压缩至 512
        self.fc2 = nn.Linear(512, num_classes)   # 输出层映射至类别空间

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播计算图逻辑"""
        # 第一阶段：Conv1 -> ReLU -> Pool
        # Tensor Shape: [B, 1, 64, 64] -> [B, 32, 32, 32]
        x = self.pool(F.relu(self.conv1(x)))

        # 第二阶段：Conv2 -> ReLU -> Pool
        # Tensor Shape: [B, 32, 32, 32] -> [B, 64, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))

        # 展平特征矩阵为一维张量，适配全连接层输入规范
        # Tensor Shape: [B, 64, 16, 16] -> [B, 16384]
        x = x.view(x.size(0), -1)

        # 隐层特征聚合与激活
        # Tensor Shape: [B, 512]
        x = F.relu(self.fc1(x))

        # 输出类别对数几率 (Logits)。损失计算环节 (CrossEntropyLoss) 内部集成 Softmax。
        # Tensor Shape: [B, 62]
        x = self.fc2(x)

        return x

    @staticmethod
    def get_class_name(class_idx: int) -> str:
        """标签反向解析映射器：将数值索引转义为实际物理类别字符"""
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
    生产环境主力 CNN 架构 (引入深度扩展与正则化机制)

    【核心架构升级】
    1. 拓扑深度增加：追加 Conv3 层，提升网络感受野与全局语义捕获能力。
    2. 批次归一化 (BatchNorm)：稳定前向特征分布，规避深度增加带来的梯度消失风险。
    3. 随机失活 (Dropout)：阻断神经元间的共适应性 (Co-adaptation)，抑制模型在训练集上的过拟合倾向。
    """
    def __init__(self, num_classes: int = 62, dropout_rate: float = 0.5):
        super(DetailedCNN, self).__init__()

        # ----- Block 1 -----
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        # 激活前应用 BatchNorm，统一特征分布的均值与方差，提升优化器收敛效率
        self.bn1 = nn.BatchNorm2d(32)

        # ----- Block 2 -----
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # ----- Block 3 -----
        # 扩展特征维度至 128，强化复杂字形结构的表征能力
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 共享的空间下采样组件
        self.pool = nn.MaxPool2d(2, 2)

        # ----- 线性分类模块 -----
        # 维度推演：64 -> (Pool1)32 -> (Pool2)16 -> (Pool3)8
        # 输出特征规模：128通道 × 8宽 × 8高 = 8192
        self.fc1 = nn.Linear(128 * 8 * 8, 512)

        # 一维特征的批量归一化
        # 【防御机制】：防止全连接层巨量的数据偏移引发过度激活，导致神经元死亡 (Dying ReLU)
        self.bn_fc1 = nn.BatchNorm1d(512)

        # 正则化层 (Dropout)
        # 训练阶段依据设定的概率 (dropout_rate) 随机阻断神经元激活，提升泛化能力。
        self.dropout = nn.Dropout(dropout_rate)

        # 输出层映射
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播过程：严格遵守 Conv -> BatchNorm -> ReLU -> Pool 范式"""
        # Block 1: 输出尺寸 32x32
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Block 2: 输出尺寸 16x16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Block 3: 输出尺寸 8x8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # 自适应展平
        x = x.view(x.size(0), -1)

        # 全连接运算管线：Linear -> BatchNorm -> ReLU -> Dropout
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # 映射至类别分布
        x = self.fc2(x)
        return x


class ResidualBlock(nn.Module):
    """
    残差块 (ResNet 的核心网络组件)

    【残差机制说明】
    针对深层网络中普遍存在的梯度消失与网络退化 (Degradation) 问题，
    残差块引入跳跃连接 (Skip Connection)，构建恒等映射 (Identity Mapping) 路径，
    确保信息与梯度的跨层无损传递。
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        # 主干网络 (Main Path)：两组标准化 3x3 卷积提取器
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 旁路映射 (Shortcut Path)：
        # 若遭遇空间维度降采样 (stride > 1) 或特征通道跃迁，必须通过 1x1 卷积执行线性投影，
        # 以确保张量维度与主干输出严格对齐，满足后续的矩阵加法约束。
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 主路计算流
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 【残差累加 (Residual Addition)】：特征矩阵与恒等映射张量执行逐元素相加
        out += self.shortcut(x)

        # 融合后执行非线性激活
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    针对 64x64 分辨率定制的轻量级残差架构。
    采用步长卷积 (Strided Convolution) 替代最大池化进行空间降采样，减少下采样过程中的信息损耗。
    """

    def __init__(self, num_classes: int = 62):
        super(ResNet, self).__init__()
        self.in_channels = 32

        # 初始特征提取干道 (Stem)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # 级联残差层组 (Stages)
        # _make_layer 依据指定参数批量实例化并堆叠 ResidualBlock。
        # 阶段切换时，利用 stride=2 触发空间分辨率减半，同时特征通道倍增。
        self.layer1 = self._make_layer(32, num_blocks=2, stride=1)  # 维持 64x64
        self.layer2 = self._make_layer(64, num_blocks=2, stride=2)  # 降采样至 32x32
        self.layer3 = self._make_layer(128, num_blocks=2, stride=2) # 降采样至 16x16
        self.layer4 = self._make_layer(256, num_blocks=2, stride=2) # 降采样至 8x8

        # 自适应平均池化机制 (AdaptiveAvgPool)
        # 强制将任意尺寸的输入特征图通过空间重采样聚合为固定的 4x4 空间维度。
        # 此机制解耦了输入图像分辨率与全连接层参数维度的强绑定关系，增强了架构的输入兼容性。
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 全局线性映射
        # 展平维度: 256通道 × 4宽 × 4高 = 4096
        self.fc = nn.Linear(256 * 1 * 1, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """层级构建工厂函数"""
        # 首个 Block 承担潜在的降维任务，后续 Block 维持特征尺度
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        in_channels = self.in_channels
        for s in strides:
            layers.append(ResidualBlock(in_channels, out_channels, s))
            in_channels = out_channels
        # 更新实例变量，确保下一层能正确接收当前层的输出通道数
        self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """网络主干执行序列"""
        out = F.relu(self.bn1(self.conv1(x))) # Stem 阶段

        # 依次穿透深层残差矩阵
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # 全局空间特征池化与维度展平
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        # 映射至目标分类空间
        out = self.fc(out)
        return out


# ============================================================
# 架构诊断与辅助验证机制
# ============================================================

def count_parameters(model: nn.Module) -> int:
    """
    计算计算图中具备梯度更新需求的可学习参数 (Trainable Parameters) 标量总和，
    提供模型空间复杂度评估依据。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_output_shape(model: nn.Module, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    动态特征维度探测器。
    注入零值张量 (Dummy Input) 模拟前向传递，提取并返回终端特征输出的张量拓扑形状。
    """
    model.eval()
    with torch.no_grad():
        dummy_input = torch.zeros(1, *input_shape)
        output = model(dummy_input)
        return output.shape[1:]


# ============================================================
# 模块内部集成测试钩子
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CNN 拓扑架构模块 - 集成测试框架实例化")
    print("=" * 60)

    # --- SimpleCNN 拓扑验证 ---
    model_s = SimpleCNN(num_classes=62)
    print("\n[SimpleCNN] 架构拓扑状态摘要：")
    print(model_s)
    print(f"[SimpleCNN] 可训练参数总量: {count_parameters(model_s):,} (基准验证级)")

    # --- DetailedCNN 拓扑验证 ---
    model_d = DetailedCNN(num_classes=62)
    print(f"\n[DetailedCNN] 可训练参数总量: {count_parameters(model_d):,} (生产环境级)")

    # --- ResNet 拓扑验证 ---
    model_r = ResNet(num_classes=62)
    print(f"\n[ResNet] 可训练参数总量: {count_parameters(model_r):,} (高维特征提取级)")

    # --- 张量维度贯穿测试 ---
    print("\n触发前向传播张量流转测试 (目标: ResNet)：")
    dummy_input = torch.randn(4, 1, 64, 64)  # 模拟 Batch_Size=4 的标准化输入流
    output = model_r(dummy_input)
    print(f"  初始注入张量形态: {dummy_input.shape}")
    print(f"  终端输出张量形态: {output.shape} (规范期望值: [4, 62])")

    # --- 标签反向解析逻辑验证 ---
    print("\n执行类别标签逆向解析校验：")
    for idx in [0, 5, 9, 10, 35, 36, 61]:
        print(f"  索引标识 {idx:2d} -> 物理特征映射: '{SimpleCNN.get_class_name(idx)}'")

    print("\n" + "=" * 60)
    print("架构自检序列执行完毕，内部逻辑状态健康！")
    print("=" * 60)