"""
辅助工具模块 (Utility Hub - 支持 64x64 网络架构)

【系统架构定位】
本模块是整个 CNN 项目的“瑞士军刀”与“透视眼镜”。它与具体的业务逻辑解耦，
专门负责处理那些通用但繁琐的底层操作，为训练、推理和数据模块提供强大的火力支援。

【核心功能板块】
1. 数学计算器：精准推演卷积/池化层的空间尺寸变化，告别“张量维度不匹配”的报错噩梦。
2. 数据可视化 (透视眼)：将冰冷的 Loss 数字转化为直观的训练曲线，将黑盒网络中的卷积核与特征图具象化。
3. 结构体检仪：一键扫描模型层级架构，统计数百万计的参数开销。
4. 算力调度台：跨平台（Apple MPS / Nvidia CUDA / CPU）自动嗅探并分配最佳硬件加速器。
"""

from typing import Tuple, List, Dict, Optional
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 【工程避坑】：强制使用 'Agg' (Anti-Grain Geometry) 渲染后端
# 原因：当代码运行在没有连接显示器的远程 Linux 服务器，或者作为后台脚本运行时，
# 如果不写这行，Matplotlib 会尝试去寻找并弹出一个图形交互窗口 (GUI)，找不到就会直接崩溃报错。
matplotlib.use('Agg')


# ============================================================
# 1. 卷积与池化尺寸计算器 (网络架构师的算盘)
# ============================================================

def calc_conv_output_size(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1
) -> int:
    """
    计算卷积层输出特征图的单边物理尺寸。

    【数学原理】：
    遵循标准的卷积尺寸推导公式：
    $O_{size} = \lfloor \frac{I_{size} + 2 \times P - D \times (K - 1) - 1}{S} \rfloor + 1$

    参数:
        input_size: 输入图像/特征图的边长 (I)
        kernel_size: 卷积核的边长 (K)
        stride: 卷积核滑动的步长 (S)
        padding: 边缘补零的圈数 (P)
        dilation: 膨胀系数，常用于空洞卷积，默认为 1 (D)
    """
    output_size = (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    return output_size


def calc_pool_output_size(
    input_size: int,
    kernel_size: int,
    stride: int = None
) -> int:
    """
    计算池化层输出的单边物理尺寸。

    【机制】：
    如果没有显式指定步长(stride)，PyTorch 默认步长等于池化核大小。
    计算公式：$O_{size} = \lfloor \frac{I_{size}}{S} \rfloor$
    """
    if stride is None:
        stride = kernel_size
    return input_size // stride


def print_size_changes():
    """
    尺寸演变推演沙盘。
    打印展示一个标准 64x64 输入在经过两次卷积和池化后的尺寸蜕变过程。
    这对于新手理解“为什么最后展平是 16384 维”极其重要。
    """
    print("=" * 60)
    print("CNN 特征图尺寸空间演变推演（基准输入 64×64）")
    print("=" * 60)

    size = 64
    print(f"[起点] 原始图像输入: {size}×{size}")

    # Conv1: 3x3 卷积核，填充 1，步长 1 -> 尺寸不变
    size = calc_conv_output_size(size, kernel_size=3, padding=1, stride=1)
    print(f" -> 经过 Conv1 (核3×3, 填充1): 维持 {size}×{size}")

    # Pool1: 2x2 最大池化 -> 尺寸直接腰斩
    size = calc_pool_output_size(size, kernel_size=2)
    print(f" -> 经过 Pool1 (核2×2): 压缩至 {size}×{size}  <-- 空间降维 (长宽减半)")

    # Conv2: 继续特征提取
    size = calc_conv_output_size(size, kernel_size=3, padding=1, stride=1)
    print(f" -> 经过 Conv2 (核3×3, 填充1): 维持 {size}×{size}")

    # Pool2: 再次池化
    size = calc_pool_output_size(size, kernel_size=2)
    print(f" -> 经过 Pool2 (核2×2): 压缩至 {size}×{size}  <-- 二次空间降维")

    print(f"\n[终点] 准备进入全连接层：")
    print(f"假设该层有 64 个通道，则展平(Flatten)后的总特征数量为: 64通道 × {size} × {size} = {64 * size * size}")
    print("=" * 60)


# ============================================================
# 2. 数据与模型可视化工具 (撕开 AI 的黑盒)
# ============================================================

def plot_training_history(
    history: Dict[str, List[float]],
    save_path: str = "training_history.png"
):
    """
    绘制并保存宏观训练生命周期曲线（Loss 与 Accuracy）。

    【核心诊断价值】：
    1. 观察两条线的间距：如果训练准确率(蓝线)一路狂飙，但验证准确率(红线)停滞不前，
       说明模型在“死记硬背”（即过拟合），需要增大 Dropout 或增加数据增强。
    2. 观察曲线平滑度：如果曲线呈剧烈锯齿状震荡，说明学习率(Learning Rate)可能设置过大。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['train_loss']) + 1)

    # 左图：损失率 (Loss) 越低越好
    ax1.plot(epochs, history['train_loss'], 'b-', label='训练集损失 (Train Loss)', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='验证集损失 (Val Loss)', linewidth=2)
    ax1.set_xlabel('训练轮次 (Epoch)', fontsize=12)
    ax1.set_ylabel('损失值 (Loss)', fontsize=12)
    ax1.set_title('模型损失收敛曲线', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 右图：准确率 (Accuracy) 越高越好
    ax2.plot(epochs, history['train_acc'], 'b-', label='训练集准确率 (Train Acc)', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='验证集准确率 (Val Acc)', linewidth=2)
    ax2.set_xlabel('训练轮次 (Epoch)', fontsize=12)
    ax2.set_ylabel('准确率 (%)', fontsize=12)
    ax2.set_title('模型准确率爬升曲线', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 训练生命周期曲线已落盘: {save_path}")


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: str = "confusion_matrix.png",
    normalize: bool = True
):
    """
    绘制混淆矩阵热力图（62类字符的全面视力表）。

    【阅读指南】：
    完美的模型会呈现一条深色的从左上到右下的对角线（真实值=预测值）。
    对角线以外的任何深色斑块，都代表模型经常“认错”的字符对（例如把大写 'O' 认成数字 '0'）。
    """
    # 归一化处理：将绝对数量转化为百分比，防止样本数量不平衡导致颜色失真
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(12, 10))
    # 使用渐变蓝 (Blues)，数值越接近 1 (即 100%) 颜色越深
    im = ax.imshow(confusion_matrix, cmap='Blues')

    # 密集设置 62 个刻度
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    ax.set_xlabel('AI 预测给出的结果 (Predicted)', fontsize=12)
    ax.set_ylabel('数据的真实身份 (Actual)', fontsize=12)
    ax.set_title('全品类混淆矩阵热力图 (Confusion Matrix)', fontsize=14)
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"🧮 混淆矩阵热力图已生成: {save_path}")


def plot_sample_predictions(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    class_names: List[str],
    save_path: str = "predictions.png",
    num_samples: int = 16
):
    """
    随机抽取样本展示实际预测结果（模型汇报专用）。
    直观展示图像长什么样、实际标签是什么、模型猜的是什么。
    """
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            # 数据流向：GPU Tensor -> CPU -> Numpy -> 剥除冗余维度 -> 反归一化
            img = images[i].squeeze().cpu().numpy()
            img = img * 0.5 + 0.5  # 将 [-1, 1] 映射回 [0, 1] 供图像渲染

            ax.imshow(img, cmap='gray')

            true_label = class_names[labels[i]]
            pred_label = class_names[predictions[i]]

            # 视觉反馈：猜对标绿，猜错标红
            color = 'green' if predictions[i] == labels[i] else 'red'
            ax.set_title(f'AI 预测: {pred_label} | 真实值: {true_label}', color=color, fontsize=10)

        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"🎯 抽样预测图集已保存: {save_path}")


def visualize_conv_filters(
    model: nn.Module,
    layer_name: str = 'conv1',
    save_path: Optional[str] = None
):
    """
    可视化指定卷积层内的“卷积核权重” (Filters / Weights)。

    【原理解析】：
    通过把卷积核本身当作小图片画出来，我们可以看到 AI 第一层是在寻找什么特征。
    通常第一层卷积核看起来像是一些边缘检测器（横线、竖线、斜线）。
    """
    # 动态抓取目标层
    conv_layer = getattr(model, layer_name)
    # 取出权重数据 (剥离计算图并放到 CPU)
    filters = conv_layer.weight.data.cpu()

    # 最大最小归一化，把无论多大范围的权重都强行压缩到 [0, 1] 区间以便绘制灰度图
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    num_filters = filters.shape[0]
    grid_size = int(np.ceil(np.sqrt(num_filters)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    fig.suptitle(f'{layer_name} 卷积核(过滤器)权重显微图', fontsize=14)

    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            ax.imshow(filters[i, 0], cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    save_name = save_path if save_path else f'{layer_name}_filters.png'
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"🔬 卷积核显微图已保存: {save_name}")


def visualize_feature_maps(
    model: nn.Module,
    image: torch.Tensor,
    layer_name: str = 'conv1',
    save_path: Optional[str] = None
):
    """
    可视化一张真实图片在穿过某一层后，被提取出的“特征图” (Feature Maps)。

    【原理解析】：
    利用 PyTorch 的 Hook (钩子) 机制，潜入网络内部，在数据流经过目标层时“拦截”并拷贝一份输出。
    你可以看到随着网络变深，图像从最初的“轮廓清晰”逐渐变成“高度抽象的马赛克团”。
    """
    model.eval()
    feature_maps = []

    # 定义“钩子”函数：一旦目标层完成前向传播，就自动触发此函数收集数据
    def hook(_module, _input, output):
        feature_maps.append(output)

    # 将钩子挂载到指定的层上
    layer = getattr(model, layer_name)
    hook_handle = layer.register_forward_hook(hook)

    # 喂入一张图，让网络跑一次，触发钩子收集证据
    with torch.no_grad():
        _ = model(image)

    # 收集完毕，立刻拆除钩子，以免影响后续正常运算
    hook_handle.remove()

    # 提取第一张图的第一个特征组
    maps = feature_maps[0][0].cpu()
    num_maps = min(maps.shape[0], 16) # 最多展示 16 张特征图
    grid_size = 4

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle(f'{layer_name} 内部特征流转状态图 (Activations)', fontsize=14)

    for i, ax in enumerate(axes.flat):
        if i < num_maps:
            # 使用 viridis (紫绿黄) 伪彩色映射，颜色越亮代表该区域特征越强烈
            ax.imshow(maps[i], cmap='viridis')
        ax.axis('off')

    plt.tight_layout()
    save_name = save_path if save_path else f'{layer_name}_feature_maps.png'
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"🧬 特征流转状态图已保存: {save_name}")


# ============================================================
# 3. 模型结构诊断工具 (架构体检仪)
# ============================================================

def count_parameters(model: nn.Module) -> int:
    """统计整个模型中所有参与梯度更新的可训练参数总和"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_by_layer(model: nn.Module) -> Dict[str, int]:
    """像做 CT 一样，拆解并统计每一层的参数量开销，找出哪里最占显存"""
    params_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_dict[name] = param.numel()
    return params_dict


def print_model_summary(model: nn.Module, input_size: Tuple[int, ...] = (1, 64, 64)):
    """
    打印详细的网络层级结构分析报告，并执行一次前向模拟验证张量尺寸是否匹配。
    """
    print("\n" + "=" * 60)
    print("AI 架构剖析摘要 (Model Architecture Summary)")
    print("=" * 60)

    total = 0
    print("\n[各层参数配额细节]:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            params = param.numel()
            total += params
            print(f"  {name}: 形状 {list(param.shape)} -> 产生 {params:,} 个参数节点")

    print(f"\n[全局汇总] 神经网络总可训练参数量: {total:,}")

    # 安全地进行一次模拟张量穿透测试，以尽早暴露尺寸崩盘的问题
    model.eval()
    with torch.no_grad():
        dummy_input = torch.zeros(1, *input_size)
        try:
            output = model(dummy_input)
            print(f"  [穿透测试] 输入张量尺寸: {list(dummy_input.shape)}")
            print(f"  [穿透测试] 输出张量尺寸: {list(output.shape)}")
            print("  [状态] 张量流通完美，未检测到尺寸断裂！")
        except Exception as e:
            print(f"\n【💥 严重架构异常警告】张量流转失败！\n请检查 CNN 卷积输出到全连接层(Linear)之间的 Flatten 维度是否对应。\n底层错误信息: {e}")

    print("=" * 60)


# ============================================================
# 4. 跨平台硬件加速管理器 (AI 引擎调度台)
# ============================================================

def get_device() -> torch.device:
    """
    自动感知宿主环境，挂载最佳的深度学习硬件加速器。
    【支持链条】：
    优先级 1. Nvidia CUDA (高端独立显卡)
    优先级 2. Apple MPS (Mac M系芯片的专属加速，Metal Performance Shaders)
    优先级 3. CPU (最后的防线)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # 【Mac M5 芯片专项优化】：如果检测到您的 MacBook Air，会稳定触发此逻辑
        return torch.device("mps")
    else:
        return torch.device("cpu")


def to_device(data, device: torch.device):
    """
    高阶张量搬运工。
    安全地将普通张量，或者嵌套在列表/元组中的多个张量，一并发送至目标硬件显存。
    """
    if isinstance(data, (list, tuple)):
        # 遇到列表，递归分解后搬运
        return [to_device(x, device) for x in data]
    return data.to(device)


# ============================================================
# 工具包自身存活性测试桩
# 当直接运行 `python src/utils.py` 时触发
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Utils 辅助工具集 - 模块自检程序启动")
    print("=" * 60)

    # 1. 测试尺寸计算沙盘
    print_size_changes()

    # 2. 临时搭建一个微型网络，测试概览器是否正常工作
    mock_model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(32 * 32 * 32, 62)
    )

    print_model_summary(mock_model, input_size=(1, 64, 64))

    # 3. 硬件嗅探测试
    current_device = get_device()
    print(f"\n⚡ 当前系统已激活的物理加速设备为: [{current_device}]")
    if str(current_device) == 'mps':
        print("  -> 检测到 Apple Silicon (Mac M系列) 硬件加速引擎已就绪！")

    print("\n" + "=" * 60)
    print("Utils 模块自检运行完毕，一切正常。")
    print("=" * 60)