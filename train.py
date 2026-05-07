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

4. SEResNet 模式  : python train.py --net seresnet
   - 特点：在 ResNet 基础上集成 SE-Block 通道注意力机制，自动学习哪些特征通道更重要。
           对易混淆字符（0/O、1/l）的区分能力更强。

💡 提示：训练执行期间或结束后，可通过终端执行 `tensorboard --logdir=logs`
启动可视化面板，以监控对比各网络结构的 Loss 与 Accuracy 演变趋势。
============================================================

[起点] 调度启动 (Command Line & Environment Setup)
  │
  ├─ 1. 资源挂载: 自动分配计算硬件 (GPU/MPS/CPU)，启动三大数据流管道 (DataLoader)
  ├─ 2. 装备组装: 实例化指定网络架构 (--net)，绑定交叉熵损失函数与 AdamW 优化器
  │
[时代大循环] for epoch in range(epochs):
  │
  ├──> [3.1 核心训练] (train_one_epoch)
  │      ├─ 状态切换: 开启 model.train() (激活 Dropout 与 BatchNorm 更新)
  │      ├─ 梯度清零: optimizer.zero_grad() (清除上一批次的残留干扰)
  │      ├─ 前向反向: 获取预测输出 -> 计算 Loss 误差 -> loss.backward() 反向求导
  │      └─ 参数更新: optimizer.step() (根据梯度微调全网神经元权重)
  │
  ├──> [3.2 独立验证] (validate)
  │      ├─ 状态切换: 开启 model.eval() 并在 torch.no_grad() 环境下运行
  │      └─ 闭卷考试: 冻结所有参数，在 Val 集上进行纯推理，得出本轮泛化得分
  │
  └──> [3.3 结算存档] (Scheduler & Checkpoint)
         ├─ 动态调优: 学习率调度器监控 Loss，遇收敛瓶颈则自动衰减学习率
         └─ 权重固化: 若验证准确率破历史记录，则保存最新参数至 best_model.pth
  │
[终点] 终局验收与产出 (Final Evaluation)
  ├──> 终极测试: 循环结束，提取历史最强存档，在绝对隔离的 Test 集上输出最终 Acc 成绩
  └──> 最终产物: TensorBoard 可视化训练曲线 + 随时可用于生产环境的推理权重

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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ============================================================
# 架构组件导入
# ============================================================
from src import FocalLoss, count_parameters, create_dataloaders, get_device
from src.models import create_model, MODEL_REGISTRY, list_models

# ============================================================
# 核心辅助函数：模型前向与反向传播逻辑
# ============================================================

def train_one_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epoch: int,
    scheduler: optim.lr_scheduler.OneCycleLR = None
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

        # OneCycleLR：每个 batch 后更新学习率（必须）
        if scheduler is not None:
            scheduler.step()

        # 统计批次误差以进行宏观监控
        running_loss += loss.item()

        # output.max(1) 返回按行求得的最大概率值及其索引序列。提取索引作为预测类别标签。
        _, predicted = output.max(1)
        total += target.size(0)

        # 统计本批次的预测命中数
        correct += predicted.eq(target).sum().item()

        # 周期性输出训练进度日志（跳过 batch 0，避免在训练开始前就打印一次无意义的状态）
        if batch_idx > 0 and batch_idx % 50 == 0:
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
    log_dir: str = "logs",
    loss_type: str = "crossentropy"
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
    model = create_model(net_type, num_classes=62).to(device)
    print(f"🧠 拓扑实例化完成: {net_type.upper()}")
    desc = list_models().get(net_type, "")
    if desc:
        print(f"   描述: {desc}")

    print(f"📊 当前模型可训练参数总量: {count_parameters(model):,}")

    # 5. 定义损失函数
    # 支持两种损失函数：
    # - crossentropy: 带 0.1 标签平滑的标准交叉熵（默认），减轻模型过度自信
    # - focal: 焦点损失，自动聚焦难例样本，适合解决 0/O、1/l 易混淆问题
    if loss_type == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print("📉 损失函数: Focal Loss (自动聚焦难例样本)")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        print("📉 损失函数: CrossEntropyLoss (label_smoothing=0.1)")

    # 优化器配置：采用 AdamW 算法，附加 weight_decay=1e-4 的 L2 正则化项以抑制权重过载膨胀。
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 动态学习率调度策略：采用 OneCycleLR 单周期学习率调度。
    # 特点：前半段学习率从低到高(热身)，后半段从高到低(余弦退火)
    # 相比 CosineAnnealingWarmRestarts，OneCycleLR 在 30-50 epoch 内通常表现更优
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,      # 前 10% 用于热身
        anneal_strategy='cos',
        div_factor=25,      # 初始学习率 = max_lr/25
        final_div_factor=1e4  # 最终学习率 = max_lr/10000
    )

    start_epoch = 1
    best_acc = 0.0

    # 6. 执行主体迭代式训练循环
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n{'='*60}\nEpoch {epoch}/{epochs}\n{'='*60}")

        # 在训练数据分布上执行参数优化
        t_loss, t_acc = train_one_epoch(model, device, train_loader, optimizer, criterion, epoch, scheduler)

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
        choices=list(MODEL_REGISTRY.keys()),
        help=f'指定网络架构: {", ".join(MODEL_REGISTRY.keys())}'
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='crossentropy',
        choices=['crossentropy', 'focal'],
        help='指定损失函数: crossentropy (默认，带标签平滑) 或 focal (聚焦难例)'
    )
    args = parser.parse_args()

    # 获取执行脚本的绝对物理路径，作为资源文件的寻址基准点，规避跨目录执行时的异常
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建统一运行时超参数及路径配置字典
    config = {
        'net_type': args.net,
        'loss_type': args.loss,
        'data_dir': os.path.join(base_dir, 'data', 'processed'),
        'epochs': 60,
        'batch_size': 128,
        'learning_rate': 0.001,
        'image_size': 64,
        'checkpoint_dir': os.path.join(base_dir, 'checkpoints'),
        'log_dir': os.path.join(base_dir, 'logs')
    }

    print("\n" + "=" * 60)
    print(f"🚀 CNN 字符识别模型训练开始 | 拓扑: {args.net.upper()} | 损失: {args.loss.upper()}")
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