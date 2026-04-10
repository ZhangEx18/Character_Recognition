"""
辅助工具模块 (支持 64x64 网络架构)

功能：
1. 绘图工具（训练曲线、混淆矩阵、预测可视化、特征图/卷积核可视化）
2. 卷积/池化层输出尺寸计算
3. 模型参数结构统计
4. 跨平台硬件加速设备检测
"""

from typing import Tuple, List, Dict, Optional
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# 使用 Agg 后端：在服务器或无图形界面的环境下（如脚本批量运行）生成图像文件，避免弹窗报错
matplotlib.use('Agg')


# ============================================================
# 卷积与池化尺寸计算器
# ============================================================

def calc_conv_output_size(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1
) -> int:
    """计算卷积层输出尺寸 (正方形图像的边长)"""
    output_size = (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    return output_size


def calc_pool_output_size(
    input_size: int,
    kernel_size: int,
    stride: int = None
) -> int:
    """计算池化层输出尺寸"""
    if stride is None:
        stride = kernel_size
    return input_size // stride


def print_size_changes():
    """打印 CNN 中特征图尺寸变化示例 (针对 64x64 架构优化)"""
    print("=" * 60)
    print("CNN 特征图尺寸变化示例（输入 64×64）")
    print("=" * 60)

    size = 64
    print(f"输入: {size}×{size}")

    size = calc_conv_output_size(size, kernel_size=3, padding=1, stride=1)
    print(f"Conv1 (3×3, pad=1): {size}×{size}")

    size = calc_pool_output_size(size, kernel_size=2)
    print(f"Pool1 (2×2): {size}×{size}  <-- 尺寸减半")

    size = calc_conv_output_size(size, kernel_size=3, padding=1, stride=1)
    print(f"Conv2 (3×3, pad=1): {size}×{size}")

    size = calc_pool_output_size(size, kernel_size=2)
    print(f"Pool2 (2×2): {size}×{size}  <-- 尺寸再次减半")

    print(f"\n展平后: 64 通道 × {size} × {size} = {64 * size * size}")
    print("=" * 60)


# ============================================================
# 数据与模型可视化工具
# ============================================================

def plot_training_history(
    history: Dict[str, List[float]],
    save_path: str = "training_history.png"
):
    """绘制并保存训练/验证过程的 Loss 与 Accuracy 曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('损失(Loss)曲线', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], 'b-', label='训练准确率', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='验证准确率', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('准确率(Accuracy)曲线', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存: {save_path}")


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: str = "confusion_matrix.png",
    normalize: bool = True
):
    """绘制混淆矩阵热力图"""
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(confusion_matrix, cmap='Blues')

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    ax.set_xlabel('预测类别 (Predicted)', fontsize=12)
    ax.set_ylabel('真实类别 (Actual)', fontsize=12)
    ax.set_title('混淆矩阵 (Confusion Matrix)', fontsize=14)
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")


def plot_sample_predictions(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    class_names: List[str],
    save_path: str = "predictions.png",
    num_samples: int = 16
):
    """抽取样本展示实际预测结果"""
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            img = images[i].squeeze().cpu().numpy()
            img = img * 0.5 + 0.5

            ax.imshow(img, cmap='gray')

            true_label = class_names[labels[i]]
            pred_label = class_names[predictions[i]]

            color = 'green' if predictions[i] == labels[i] else 'red'
            ax.set_title(f'预测: {pred_label} | 真实: {true_label}', color=color, fontsize=10)

        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"预测结果已保存: {save_path}")


def visualize_conv_filters(
    model: nn.Module,
    layer_name: str = 'conv1',
    save_path: Optional[str] = None
):
    """可视化网络某层的卷积核权重"""
    conv_layer = getattr(model, layer_name)
    filters = conv_layer.weight.data.cpu()

    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    num_filters = filters.shape[0]
    grid_size = int(np.ceil(np.sqrt(num_filters)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    fig.suptitle(f'{layer_name} 卷积核可视化 (Weights)', fontsize=14)

    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            ax.imshow(filters[i, 0], cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    save_name = save_path if save_path else f'{layer_name}_filters.png'
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"卷积核可视化已保存: {save_name}")


def visualize_feature_maps(
    model: nn.Module,
    image: torch.Tensor,
    layer_name: str = 'conv1',
    save_path: Optional[str] = None
):
    """可视化图片经过某一层后生成的 特征图"""
    model.eval()
    feature_maps = []

    # 使用下划线前缀，解决未使用参数和名字冲突警告
    def hook(_module, _input, output):
        feature_maps.append(output)

    layer = getattr(model, layer_name)
    hook_handle = layer.register_forward_hook(hook)

    with torch.no_grad():
        _ = model(image)

    hook_handle.remove()

    maps = feature_maps[0][0].cpu()
    num_maps = min(maps.shape[0], 16)
    grid_size = 4

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle(f'{layer_name} 特征图可视化 (Activations)', fontsize=14)

    for i, ax in enumerate(axes.flat):
        if i < num_maps:
            ax.imshow(maps[i], cmap='viridis')
        ax.axis('off')

    plt.tight_layout()
    save_name = save_path if save_path else f'{layer_name}_feature_maps.png'
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"特征图可视化已保存: {save_name}")


# ============================================================
# 模型结构诊断工具
# ============================================================

def count_parameters(model: nn.Module) -> int:
    """统计模型中需要梯度更新的总参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_by_layer(model: nn.Module) -> Dict[str, int]:
    """拆解统计各网络层的参数量"""
    params_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_dict[name] = param.numel()
    return params_dict


def print_model_summary(model: nn.Module, input_size: Tuple[int, ...] = (1, 64, 64)):
    """打印网络概览信息"""
    print("\n" + "=" * 60)
    print("模型摘要 (Model Summary)")
    print("=" * 60)

    total = 0
    print("\n各层参数细节:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            params = param.numel()
            total += params
            print(f"  {name}: {list(param.shape)} -> 产生 {params:,} 个参数")

    print(f"\n总可训练参数量: {total:,}")

    model.eval()
    with torch.no_grad():
        dummy_input = torch.zeros(1, *input_size)
        try:
            output = model(dummy_input)
            print(f"测试输入形状: {list(dummy_input.shape)}")
            print(f"测试输出形状: {list(output.shape)}")
        except Exception as e:
            print(f"【尺寸异常警告】张量流转失败，请检查网络架构设计！\n错误信息: {e}")

    print("=" * 60)


# ============================================================
# 跨平台硬件加速管理
# ============================================================

def get_device() -> torch.device:
    """自动检测可用的最佳硬件设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def to_device(data, device: torch.device):
    """将张量或张量集合安全地移交至指定设备"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)


# ============================================================
# 工具包自身测试桩
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("工具函数模块初始化测试")
    print("=" * 60)

    print_size_changes()

    mock_model = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(32 * 32 * 32, 62)
    )

    print_model_summary(mock_model, input_size=(1, 64, 64))

    # 解决外部作用域名称隐藏警告
    current_device = get_device()
    print(f"\n当前系统已激活的加速设备为: {current_device}")

    print("\n" + "=" * 60)
    print("辅助工具模块测试完毕!")
    print("=" * 60)