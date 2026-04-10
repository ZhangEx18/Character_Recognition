"""
主训练入口脚本 (多模型动态切换版)

============================================================
【快速启动指南】
通过终端的 --net 参数，您可以随心所欲地切换三种不同的模型架构：

1. 快速验证模式 (SimpleCNN)  : python train.py --net simple
   - 特点：层数最少，速度极快，适合用来跑通数据流或快速找 Bug。

2. 主力训练模式 (DetailedCNN): python train.py --net detailed
   - 特点：默认选项。带有 BatchNorm 和 Dropout，性能稳健，是交付的主力。

3. 深度残差模式 (ResNet)     : python train.py --net resnet
   - 特点：针对 64x64 定制的轻量级残差网络。结构最深，潜力最大。

💡 提示：运行结束后，在终端输入 `tensorboard --logdir=logs`，
即可在浏览器中直观对比这三个模型在同一张图表上的准确率变化曲线！
============================================================

【系统功能总览】：
1. 硬件自适应：自动嗅探并完美调度 Mac MPS (Apple Silicon)、CUDA 或 CPU。
2. 路径免疫机制：动态计算工程绝对路径，彻底告别 FileNotFoundError 烦恼。
3. 严格验证隔离：每个 Epoch 独立进行无梯度的验证集测试，严防数据泄露。
4. 状态持久化：自动拦截并保存拥有最高准确率的 Checkpoint，支持断点续训。
5. 动态学习率：搭载 ReduceLROnPlateau，在 Loss 进入瓶颈期时自动收缩学习率。
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
# 从核心包 (src) 导入架构组件
# ============================================================
from src import DetailedCNN, SimpleCNN, ResNet, count_parameters, create_dataloaders, get_device

# ============================================================
# 核心辅助函数：训练与验证的底层逻辑
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
    执行单个 Epoch (全量数据遍历一次) 的训练过程。

    【核心机制】：
    在这个阶段，模型处于“学习状态”，所有的权重参数都会根据损失函数的反馈进行调整。
    """
    # 1. 切换为训练模式。这非常重要，它会激活 Dropout 层（随机丢弃神经元）和
    # BatchNorm 层（动态计算当前批次的均值和方差），防止模型死记硬背。
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    # 遍历数据加载器，每次吐出一小批 (Batch，例如 128 张) 图片和对应的正确答案
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据搬运到指定的加速硬件上 (如 Mac 的 MPS)
        data = data.to(device)
        target = target.to(device)

        # 【PyTorch 黄金五步法则】
        # 第 1 步：清空历史梯度。PyTorch 默认会累加梯度，如果不清零，
        # 前几次的计算结果会干扰当前批次的更新方向。
        optimizer.zero_grad()

        # 第 2 步：前向传播 (Forward)。让模型看着图片猜答案。
        output = model(data)

        # 第 3 步：计算损失 (Loss)。用交叉熵函数对比“模型猜的”和“标准答案”之间的差距。
        loss = criterion(output, target)

        # 第 4 步：反向传播 (Backward)。利用高等代数的链式法则，计算出模型中
        # 数百万个参数对当前误差的“责任大小”（即梯度）。
        loss.backward()

        # 第 5 步：权重更新 (Step)。优化器 (如 AdamW) 根据刚才算出的梯度，
        # 稍微调整一下模型的内部参数，让下一次猜得更准一点。
        optimizer.step()

        # 记录统计数据用于日志打印
        running_loss += loss.item()

        # output.max(1) 返回每一行最大概率的值及其索引。我们只需要索引（即预测的类别标签）
        _, predicted = output.max(1)
        total += target.size(0)

        # 统计本批次中猜对的数量 (.eq 就是 equal 判断，.sum 求和)
        correct += predicted.eq(target).sum().item()

        # 每隔 50 个批次打印一次进度报告，防止终端没有响应让人以为死机了
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
    在验证集或测试集上执行结业考试，绝对不更新任何权重。

    【核心机制】：
    在这个阶段，模型处于“考试状态”，只能调用已有知识作答，不允许翻书学习（更新参数）。
    """
    # 1. 切换为评估模式。这会冻结 Dropout (不再丢弃神经元) 和
    # BatchNorm (使用训练时积累的全局均值和方差)，确保每次考试的条件绝对公平稳定。
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    # 2. 【极其关键】：上下文管理器 torch.no_grad()
    # 强制告诉 PyTorch：“接下来发生的所有事情都不要记录计算图”。
    # 这会直接节省 50% 以上的显存空间，并大幅提升测试速度。
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
    保存包含优化器状态的全量检查点 (Checkpoint)，支持意外中断后的“断点续训”。

    【为什么要存 Optimizer？】
    Adam 等高级优化器内部维护着动量 (Momentum) 和方差等历史状态。如果只存模型权重，
    中途断电后再恢复训练时，优化器相当于被“洗脑”了，会导致 Loss 突然飙升。
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 将所有必要的状态打包成一个字典
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),           # 模型的骨架和肌肉(权重参数)
        'optimizer_state_dict': optimizer.state_dict(),   # 优化器的记忆(动量等)
        'best_acc': best_acc                              # 当前的最佳成绩
    }

    path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, path)
    print(f"安全快照已存档: {path}")


# ============================================================
# 主训练调度逻辑 (Pipeline)
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
    """执行并统筹完整的模型训练生命周期"""

    # 1. 硬件准备
    device = get_device()

    # 2. TensorBoard 日志记录仪准备
    # 为每次运行生成带有时间戳的独立日志文件夹，防止被后续运行覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(log_dir, f"{net_type}_{timestamp}"))

    # 3. 吊起数据流水线，获取打乱的训练流、纯净的验证流和测试流
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size
    )

    # 4. 根据用户的指令挂载对应的模型“大脑”
    if net_type == 'simple':
        model = SimpleCNN(num_classes=62).to(device)
        print("🔧 已挂载架构: SimpleCNN (简单卷积，适合快速验证)")
    elif net_type == 'detailed':
        model = DetailedCNN(num_classes=62).to(device)
        print("⚙️ 已挂载架构: DetailedCNN (带 BatchNorm 的加强版卷积)")
    elif net_type == 'resnet':
        model = ResNet(num_classes=62).to(device)
        print("🧠 已挂载架构: ResNet (深层残差网络，潜力最大)")
    else:
        raise ValueError(f"未知的网络类型: {net_type}")

    print(f"📊 模型总可训练参数量: {count_parameters(model):,}")

    # 5. 设定监考官与营养师
    # 监考官 (损失函数)：使用自带 0.1 标签平滑的交叉熵。
    # 标签平滑会让模型即使做对了也不会给出 100% 的过度自信，能有效防止过拟合。
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 营养师 (优化器)：使用 AdamW 优化算法，并加入轻微的权重衰减 (weight_decay=1e-4) 防止模型参数过大。
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 动态调参仪 (调度器)：当验证集 Loss 连续 5 次 (patience) 没有下降时，
    # 自动将学习率砍半 (factor=0.5)，帮助模型进行微调，跳出局部的“浅坑”。
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    start_epoch = 1
    best_acc = 0.0

    # 6. 正式开始按轮次 (Epoch) 循环训练
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n{'='*60}\nEpoch {epoch}/{epochs}\n{'='*60}")

        # 让模型去训练集学习，拿到平时的模拟考成绩
        t_loss, t_acc = train_one_epoch(model, device, train_loader, optimizer, criterion, epoch)

        # 让模型去验证集考试，拿到严格不透题的成绩
        v_loss, v_acc = validate(model, device, val_loader, criterion)

        # 将所有的核心指标打点记录到 TensorBoard，用于后续可视化画图
        writer.add_scalar('Loss/train', t_loss, epoch)
        writer.add_scalar('Loss/val', v_loss, epoch)
        writer.add_scalar('Accuracy/train', t_acc, epoch)
        writer.add_scalar('Accuracy/val', v_acc, epoch)
        # 记录当前的学习率大小，方便观察 scheduler 是否触发了衰减
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

        print(f"\n总结: Train Acc {t_acc:.2f}% | Val Acc {v_acc:.2f}% | LR {optimizer.param_groups[0]['lr']:.6f}")

        # 让调度器根据这次考试的 Loss 决定要不要收缩学习率
        scheduler.step(v_loss)

        # 7. 优胜劣汰机制 (模型保存)
        # 如果这次考试的准确率打破了历史最高记录，立即封存这个版本的模型为 "best_model.pth"
        if v_acc > best_acc:
            best_acc = v_acc
            save_checkpoint(model, optimizer, epoch, best_acc, checkpoint_dir)

            # 这个是不带优化器状态的轻量级版本，专门留给推理端 (inference/camera) 使用
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"  ★ 性能突破！最佳准确率已更新: {best_acc:.2f}%")

    # 8. 训练彻底结束后的终极盲测
    # 重新加载我们在训练过程中保存的那个“巅峰状态”的模型权重
    best_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    # 去拿测试集 (Test Set) 进行最终测验。
    # 之前训练时模型从未见过这些图片，哪怕是在验证过程中。这是真正检验实力的时刻。
    final_test_loss, final_test_acc = validate(model, device, test_loader, criterion)
    print(f"\n{'='*60}\n训练流水线彻底结束 | 最终系统盲测准确率: {final_test_acc:.2f}%\n{'='*60}")

    # 关闭日志流
    writer.close()


# ============================================================
# CLI 命令行入口保护块
# ============================================================

def main():
    """解析终端传入的参数并启动系统调度核心"""
    parser = argparse.ArgumentParser(description='CNN 多模型自由切换训练脚本')

    parser.add_argument(
        '--net',
        type=str,
        default='detailed',
        choices=['simple', 'detailed', 'resnet'],
        help='选择要训练的网络架构: simple, detailed, 或 resnet'
    )
    args = parser.parse_args()

    # 获取当前执行脚本所在的绝对路径，确保在任何终端目录下运行都不会报“找不到文件”错误
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 组装超级配置字典
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
    print(f"🚀 CNN 字符识别系统 - 训练启动台 | 当前模式: {args.net.upper()}")
    print("=" * 60)

    # 目录健康度检查
    if not os.path.exists(config['data_dir']):
        print(f"\n[致命错误]: 未找到预处理后的数据 -> {config['data_dir']}")
        print("提示：请先运行 tools/preprocess.py")
        return

    # 【工程修复】：在此使用 ** 进行字典解包 (Dictionary Unpacking)。
    # 它等价于自动将字典拆解为: train(net_type='xxx', data_dir='xxx', epochs=30 ...)
    # 这比写一长串参数传递要优雅和健壮得多。
    train(**config)


if __name__ == "__main__":
    main()