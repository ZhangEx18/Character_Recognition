"""
主训练入口脚本 (多模型架构配置版)

============================================================
【参数配置与执行说明】
通过命令行参数 `--net` 可指定实例化的模型架构：

1. SimpleCNN 模式 : python train.py --net simple
   - 特点：浅层卷积神经网络，参数量极小，适用于数据流测试与基准验证。

2. DetailedCNN 模式: python train.py --net detailed
   - 特点：默认选项。引入 BatchNorm 与 Dropout 层的标准卷积网络，提供稳健的收敛性能。

3. ResNet 模式    : python train.py --net resnet
   - 特点：针对 64x64 输入分辨率适配的轻量级残差网络。网络层数较深，特征提取能力更强。

💡 提示：训练执行期间或结束后，可通过终端执行 `tensorboard --logdir=logs`
启动可视化面板，以监控对比各网络结构的 Loss 与 Accuracy 演变趋势。
============================================================

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

============================================================

【系统核心模块总览】：
1. 硬件加速适配：自动检测并分配至 MPS (Apple Silicon)、CUDA 或 CPU 计算设备。
2. 动态路径寻址：基于脚本所在目录计算工程绝对路径，避免执行路径差异导致的 I/O 异常。
3. 验证隔离机制：每个 Epoch 执行无梯度的独立验证阶段，严防训练数据泄露与过拟合。
4. 状态持久化机制：依据验证集评价指标自动保存最佳模型权重，并保留优化器状态以支持断点续训。
5. 学习率动态衰减：集成 ReduceLROnPlateau 调度器，在损失函数收敛停滞时自动按比例降低学习率。

"""

import argparse
import os
from pathlib import Path
from typing import Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ============================================================
# 架构组件导入
# ============================================================
from src import DetailedCNN, SimpleCNN, ResNet, count_parameters, create_dataloaders, get_device

# ============================================================
# 核心辅助函数：模型前向与反向传播逻辑
# ============================================================

def train_one_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epoch: int
) -> Tuple[float, float]:
    """
    执行单个 Epoch (全量训练数据完整遍历一次) 的训练迭代。

    【核心机制】：
    在该阶段，模型计算前向传播误差，并通过反向传播算法更新网络权重参数。
    """
    # 1. 切换至训练模式。此操作将启用 Dropout 层的随机失活机制，
    # 并激活 BatchNorm 层的批次均值和方差统计更新，从而提升模型泛化能力。
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    # 遍历训练数据加载器，按 Batch (例如 128 个样本) 获取输入图像与真实标签
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将张量迁移至指定的加速设备
        data = data.to(device)
        target = target.to(device)

        # 【标准参数优化流程】
        # 步骤 1：梯度清零。清除前序批次的累积梯度，防止干扰当前优化的计算方向。
        optimizer.zero_grad()

        # 步骤 2：前向传播 (Forward pass)。获取当前输入在模型下的预测分布。
        output = model(data)

        # 步骤 3：损失计算 (Loss calculation)。通过目标函数量化预测分布与真实标签的散度。
        loss = criterion(output, target)

        # 步骤 4：反向传播 (Backward pass)。依据链式法则，计算损失函数对各可学习参数的梯度。
        loss.backward()

        # 步骤 5：参数更新 (Optimization step)。优化算法基于当前梯度调整网络权重。
        optimizer.step()

        # 统计批次误差以进行宏观监控
        running_loss += loss.item()

        # output.max(1) 返回按行求得的最大概率值及其索引序列。提取索引作为预测类别标签。
        _, predicted = output.max(1)
        total += target.size(0)

        # 统计本批次的预测命中数
        correct += predicted.eq(target).sum().item()

        # 周期性输出训练进度日志，监控批次层面的收敛状态
        if batch_idx % 50 == 0:
            acc = 100. * correct / total
            print(f"  Epoch [{epoch}] | Batch {batch_idx:3d}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | Acc: {acc:.2f}%")

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    device: torch.device,
    val_loader: DataLoader,
    criterion: nn.Module
) -> Tuple[float, float]:
    """
    在验证集或测试集上执行模型评估。

    【核心机制】：
    在此阶段，冻结所有网络层参数并关闭梯度计算，仅评估当前参数空间分布下的泛化性能。
    """
    # 1. 切换至评估模式。此操作将禁用 Dropout 层，并强制 BatchNorm
    # 使用训练阶段积累的全局移动平均统计量，保证验证过程的确定性。
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    # 2. 【显存与计算优化】：使用 torch.no_grad() 上下文管理器。
    # 屏蔽自动求导引擎 (Autograd)，避免构建计算图，显著降低 VRAM 占用并加速前向推理。
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_acc: float,
    save_dir: str
):
    """
    序列化保存包含优化器状态的完整检查点 (Checkpoint)，支持意外中断后的无缝恢复。

    【机制说明】：
    Adam/AdamW 等自适应优化器内部维护有动量 (Momentum) 与方差等历史一阶/二阶矩估计状态。
    联合保存优化器状态可避免中断恢复时因状态丢失而导致训练前期的 Loss 剧烈震荡。
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 构建持久化状态字典
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),           # 神经网络层的参数张量
        'optimizer_state_dict': optimizer.state_dict(),   # 优化器的动量与内部变量状态
        'best_acc': best_acc                              # 历史监控最优准确率
    }

    path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, path)
    print(f"检查点序列化完成: {path}")


# ============================================================
# 主训练调度逻辑 (Training Pipeline)
# ============================================================

def train(
    net_type: str,
    data_dir: str,
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    image_size: int = 64,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs"
):
    """协调并执行完整的神经网络训练生命周期"""

    # 1. 硬件资源分配
    device = get_device()

    # 2. 标量日志记录系统初始化
    # 基于时间戳生成独立目录，避免多次运行导致的 TensorBoard 事件文件覆写冲突
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(log_dir, f"{net_type}_{timestamp}"))

    # 3. 初始化数据加载流水线，获取训练集、验证集和独立测试集对应的 DataLoader
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size
    )

    # 4. 根据输入配置实例化指定的网络拓扑结构
    if net_type == 'simple':
        model = SimpleCNN(num_classes=62).to(device)
        print("🔧 拓扑实例化完成: SimpleCNN (基准级网络)")
    elif net_type == 'detailed':
        model = DetailedCNN(num_classes=62).to(device)
        print("⚙️ 拓扑实例化完成: DetailedCNN (引入 BatchNorm 等稳健性结构)")
    elif net_type == 'resnet':
        model = ResNet(num_classes=62).to(device)
        print("🧠 拓扑实例化完成: ResNet (深层残差架构)")
    else:
        raise ValueError(f"不受支持的网络架构类型: {net_type}")

    print(f"📊 当前模型可训练参数总量: {count_parameters(model):,}")

    # 5. 定义损失函数与优化器参数
    # 损失函数配置：采用带 0.1 标签平滑 (Label Smoothing) 的交叉熵损失。
    # 标签平滑通过软化目标分布，可有效减轻模型预测过度自信 (Overconfidence)，提升正则化效果。
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 优化器配置：采用 AdamW 算法，附加 weight_decay=1e-4 的 L2 正则化项以抑制权重过载膨胀。
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 动态学习率调度策略：监控验证集 Loss。若连续 5 个 Epoch 性能未见改善 (patience=5)，
    # 则对当前学习率执行系数为 0.5 的收缩，促使模型在局部最优点附近进行精细收敛。
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    start_epoch = 1
    best_acc = 0.0

    # 6. 执行主体迭代式训练循环
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n{'='*60}\nEpoch {epoch}/{epochs}\n{'='*60}")

        # 在训练数据分布上执行参数优化
        t_loss, t_acc = train_one_epoch(model, device, train_loader, optimizer, criterion, epoch)

        # 在同源但相互隔离的验证集上进行泛化误差评估
        v_loss, v_acc = validate(model, device, val_loader, criterion)

        # 向 TensorBoard 标量收集器写入当前周期的核心评估指标
        writer.add_scalar('Loss/train', t_loss, epoch)
        writer.add_scalar('Loss/val', v_loss, epoch)
        writer.add_scalar('Accuracy/train', t_acc, epoch)
        writer.add_scalar('Accuracy/val', v_acc, epoch)
        # 同步记录当前实际生效的学习率，用于分析 scheduler 的调度频次
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

        print(f"\n周期摘要: Train Acc {t_acc:.2f}% | Val Acc {v_acc:.2f}% | LR {optimizer.param_groups[0]['lr']:.6f}")

        # 调度器依据本轮验证集损失指标评估是否需要触发学习率衰减
        scheduler.step(v_loss)

        # 7. 模型检查点保存策略
        # 判断当前轮次验证准确率是否突破历史阈值上限，若是，则执行状态缓存
        if v_acc > best_acc:
            best_acc = v_acc
            save_checkpoint(model, optimizer, epoch, best_acc, checkpoint_dir)

            # 导出不含优化器状态的精简权重副本，专门供推理环境 (Inference) 加载使用
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"  ★ 全局极值更新: 新的最优验证集准确率达 {best_acc:.2f}%")

    # 8. 终期独立测试集评估
    # 加载系统在历史训练轨迹中截获的具备最强泛化表现的模型权重副本
    best_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    # 在未参与模型任何形式训练或超参数微调的独立测试集上进行最终评价
    final_test_loss, final_test_acc = validate(model, device, test_loader, criterion)
    print(f"\n{'='*60}\n流水线处理完毕 | 模型独立测试集泛化准确率 (Test-Set Acc): {final_test_acc:.2f}%\n{'='*60}")

    # 刷新并关闭事件写入句柄
    writer.close()


# ============================================================
# CLI 入口与配置解析块
# ============================================================

def main():
    """解析命令行参数，构建超参数配置并启动主训练调度流程"""
    parser = argparse.ArgumentParser(description='卷积神经网络拓扑结构动态评估入口')

    parser.add_argument(
        '--net',
        type=str,
        default='detailed',
        choices=['simple', 'detailed', 'resnet'],
        help='指定实例化的网络架构类别: simple, detailed, 或 resnet'
    )
    args = parser.parse_args()

    # 获取执行脚本的绝对物理路径，作为资源文件的寻址基准点，规避跨目录执行时的异常
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建统一运行时超参数及路径配置字典
    config = {
        'net_type': args.net,
        'data_dir': os.path.join(base_dir, 'data', 'processed'),
        'epochs': 30,
        'batch_size': 128,
        'learning_rate': 0.001,
        'image_size': 64,
        'checkpoint_dir': os.path.join(base_dir, 'checkpoints'),
        'log_dir': os.path.join(base_dir, 'logs')
    }

    print("\n" + "=" * 60)
    print(f"🚀 CNN 字符识别模型训练开始 | 目标配置拓扑: {args.net.upper()}")
    print("=" * 60)

    # 路径级联依赖检查
    if not os.path.exists(config['data_dir']):
        print(f"\n[Error]: 未识别到预处理数据结构 -> {config['data_dir']}")
        print("提示：在执行训练流水线前，需确保已完成 tools/preprocess.py 数据清理过程。")
        return

    # 【字典解包参数传递】：使用 ** 语法自动展开 config 字典对应的键值对。
    # 该方式可作为关键字参数传递给目标函数，相较于冗长的显式赋值，具备更高的代码拓展性。
    train(**config)


if __name__ == "__main__":
    main()