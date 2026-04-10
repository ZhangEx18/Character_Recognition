"""
主训练入口脚本 (放置于项目根目录)
运行方式: python train.py

功能总览：
1. 训练循环：自动调度已检测到的 CUDA / MPS / CPU 硬件加速。
2. 路径自适应：动态获取脚本所在位置，确保在任何环境下都能正确挂载数据。
3. 验证与评估：在每个 Epoch 结束后执行不产生梯度的验证集评估。
4. 状态持久化：记录 TensorBoard 日志，并自动保存验证集准确率最高的模型。
5. 动态学习率：监控验证集 Loss，在进入瓶颈期时自动收缩学习率。
"""

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
from src import DetailedCNN, count_parameters, create_dataloaders, get_device


def train_one_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epoch: int
) -> Tuple[float, float]:
    """
    执行单个 Epoch 的训练过程

    参数:
        model: 神经网络模型
        device: 计算设备 (GPU/MPS/CPU)
        train_loader: 训练数据加载器
        optimizer: 优化器 (如 Adam)
        criterion: 损失函数 (如 CrossEntropyLoss)
        epoch: 当前轮次
    """
    # 切换至训练模式：启用 Dropout 和 BatchNorm 的运行统计
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # 数据移交至加速硬件
        data = data.to(device)
        target = target.to(device)

        # --- 训练核心五步曲 ---
        # 1. 梯度清零：防止上一个 batch 的梯度干扰当前计算
        optimizer.zero_grad()

        # 2. 前向传播：计算预测得分
        output = model(data)

        # 3. 计算损失：衡量预测值与真实标签的差异
        loss = criterion(output, target)

        # 4. 反向传播：根据损失计算每个权重的导数 (Gradient)
        loss.backward()

        # 5. 更新权重：优化器根据梯度调整模型参数
        optimizer.step()

        # 统计进度
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # 降频打印进度，减少 I/O 损耗
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
    # 切换至评估模式：锁定 BatchNorm 均值，关闭 Dropout
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    # 禁用梯度计算：大幅降低内存消耗，提升计算速度
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


def train(
    data_dir: str,
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    image_size: int = 64,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    resume: str = None
):
    """执行完整的训练流水线"""

    # 1. 设备与环境初始化
    device = get_device()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(log_dir, timestamp))

    # 2. 构建数据供血系统
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size
    )

    # 3. 实例化模型与优化器
    model = DetailedCNN(num_classes=62).to(device)
    print(f"\n模型已就绪，可训练参数总量: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 动态学习率策略：连续 5 轮验证集损失不下降，则学习率减半
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    start_epoch = 1
    best_acc = 0.0

    # 4. 训练大循环
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n{'='*60}\nEpoch {epoch}/{epochs}\n{'='*60}")

        # A. 训练阶段
        t_loss, t_acc = train_one_epoch(model, device, train_loader, optimizer, criterion, epoch)

        # B. 验证阶段
        v_loss, v_acc = validate(model, device, val_loader, criterion)

        # C. 记录日志
        writer.add_scalar('Loss/train', t_loss, epoch)
        writer.add_scalar('Loss/val', v_loss, epoch)
        writer.add_scalar('Accuracy/train', t_acc, epoch)
        writer.add_scalar('Accuracy/val', v_acc, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

        print(f"\n总结: Train Acc {t_acc:.2f}% | Val Acc {v_acc:.2f}% | LR {optimizer.param_groups[0]['lr']:.6f}")

        # D. 学习率调整与最佳模型筛选
        scheduler.step(v_loss)

        if v_acc > best_acc:
            best_acc = v_acc
            save_checkpoint(model, optimizer, epoch, best_acc, checkpoint_dir)
            # 额外保存一份纯权重版
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"  ★ 性能突破！最佳准确率已更新: {best_acc:.2f}%")

    # 5. 最终测试评估
    best_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    final_test_loss, final_test_acc = validate(model, device, test_loader, criterion)
    print(f"\n{'='*60}\n训练结束 | 最终盲测准确率: {final_test_acc:.2f}%\n{'='*60}")

    writer.close()


def main():
    """解析路径并启动任务"""

    # 动态获取当前 train.py 所在的绝对路径，确保相对路径 data/processed 始终有效
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    config = {
        'data_dir': os.path.join(BASE_DIR, 'data', 'processed'),
        'epochs': 30,
        'batch_size': 128,
        'learning_rate': 0.001,
        'image_size': 64,
        'checkpoint_dir': os.path.join(BASE_DIR, 'checkpoints'),
        'log_dir': os.path.join(BASE_DIR, 'logs')
    }

    print("\n" + "=" * 60)
    print("CNN 字符识别系统 - 训练启动台")
    print("=" * 60)

    if not os.path.exists(config['data_dir']):
        print(f"\n[致命错误]: 未找到预处理后的数据 -> {config['data_dir']}")
        print("💡 请先运行: python -m tools.preprocess")
        return

    # 启动训练
    train(**config)


if __name__ == "__main__":
    main()