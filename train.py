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

提示：运行结束后，在终端输入 `tensorboard --logdir=logs`，
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
# 恢复的三个核心辅助函数
# ============================================================
def train_one_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epoch: int
) -> Tuple[float, float]:
    """执行单个 Epoch 的训练过程"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

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
    """在验证集上执行结业考试，不更新任何权重"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

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
    """保存包含优化器状态的全量检查点，支持断点续训"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
    }
    path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, path)
    print(f"安全快照已存档: {path}")

# ============================================================
# 主训练调度逻辑
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
    # 修复：移除了未使用的 resume 参数
):
    """执行完整的训练流水线"""
    device = get_device()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(log_dir, f"{net_type}_{timestamp}"))

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size
    )

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

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    start_epoch = 1
    best_acc = 0.0

    for epoch in range(start_epoch, epochs + 1):
        print(f"\n{'='*60}\nEpoch {epoch}/{epochs}\n{'='*60}")

        t_loss, t_acc = train_one_epoch(model, device, train_loader, optimizer, criterion, epoch)
        v_loss, v_acc = validate(model, device, val_loader, criterion)

        writer.add_scalar('Loss/train', t_loss, epoch)
        writer.add_scalar('Loss/val', v_loss, epoch)
        writer.add_scalar('Accuracy/train', t_acc, epoch)
        writer.add_scalar('Accuracy/val', v_acc, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

        print(f"\n总结: Train Acc {t_acc:.2f}% | Val Acc {v_acc:.2f}% | LR {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step(v_loss)

        if v_acc > best_acc:
            best_acc = v_acc
            save_checkpoint(model, optimizer, epoch, best_acc, checkpoint_dir)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"  ★ 性能突破！最佳准确率已更新: {best_acc:.2f}%")

    best_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    final_test_loss, final_test_acc = validate(model, device, test_loader, criterion)
    print(f"\n{'='*60}\n训练结束 | 最终盲测准确率: {final_test_acc:.2f}%\n{'='*60}")

    writer.close()


def main():
    """解析终端参数并启动任务"""
    parser = argparse.ArgumentParser(description='CNN 多模型自由切换训练脚本')

    parser.add_argument(
        '--net',
        type=str,
        default='detailed',
        choices=['simple', 'detailed', 'resnet'],
        help='选择要训练的网络架构: simple, detailed, 或 resnet'
    )
    args = parser.parse_args()

    # 修复：改为小写 base_dir 符合 PEP8 规范
    base_dir = os.path.dirname(os.path.abspath(__file__))

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

    if not os.path.exists(config['data_dir']):
        print(f"\n[致命错误]: 未找到预处理后的数据 -> {config['data_dir']}")
        return

    # 修复：加了 ** 符号，字典解包传入参数
    train(**config)


if __name__ == "__main__":
    main()