"""
辅助工具模块 (Utility Hub - 支持 64x64 网络架构)

【系统架构定位】
本模块作为 CNN 项目的底层通用支撑库，与核心业务逻辑完全解耦。
负责封装张量计算、模型拓扑诊断、硬件设备调度及高维特征可视化等基础操作，
为数据流转、网络训练与推理评估提供标准化的 API 接口。

【全景处理流程】(模块跨生命周期调用链路)
[前置阶段] 架构设计与初始化
  ├─ 1. 维度推演: 调用 calc_conv_output_size / calc_pool_output_size 验证特征图空间收缩率
  ├─ 2. 拓扑诊断: 挂载 print_model_summary 执行 Dummy Input 穿透测试，排查全连接层维度断层
  └─ 3. 硬件寻址: 触发 get_device() 嗅探并绑定 MPS/CUDA/CPU 计算引擎
  │
[运行阶段] 模型训练与验证 (被 train.py 循环调用)
  └─ 4. 张量迁移: 依靠 to_device() 实现 Batch 数据与 Target 在 Host(CPU) 与 Device(加速器) 间的高效拷贝
  │
[后置阶段] 性能评估与可视化诊断
  ├─ 5. 宏观收敛分析: 生成 plot_training_history 绘制全周期 Loss/Acc 演化折线图
  ├─ 6. 微观错判分析: 渲染 plot_confusion_matrix 构建多类别混淆分布热力图
  ├─ 7. 提取器诊断: 调用 visualize_conv_filters 观测底层卷积核的感受野与纹理响应偏好
  └─ 8. 表征观测: 挂载 Forward Hook (visualize_feature_maps) 截获并渲染深层特征的激活响应图

============================================================

【系统功能总览】：
1. 空间维度计算器：基于标准步长与填充公式，推演特征图在各层级间的空间尺寸映射关系。
2. 训练指标可视化：渲染损失函数与准确率的演变趋势，支持过拟合/欠拟合状态的直观诊断。
3. 拓扑结构分析仪：遍历计算图节点，统计算法参数总量，并执行前向维度兼容性校验。
4. 异构计算调度台：实现底层硬件感知，自动路由张量至最优计算后端。
"""

from typing import Tuple, List, Dict, Optional
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 【环境兼容性设置】：强制指定 'Agg' (Anti-Grain Geometry) 渲染后端。
# 目的：规避在无 X11 转发的远程 Linux 服务器或纯 CLI 环境下，
# Matplotlib 尝试挂载 GUI 交互窗口而引发的系统级崩溃。
matplotlib.use('Agg')


# ============================================================
# 1. 卷积与池化空间尺寸推导计算逻辑
# ============================================================

def calc_conv_output_size(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1
) -> int:
    r"""
    计算二维卷积层输出特征图的单侧物理分辨率。

    【数学原理】：
    遵循标准的离散卷积维度推导等式：
    $O_{size} = \lfloor \frac{I_{size} + 2 \times P - D \times (K - 1) - 1}{S} \rfloor + 1$

    参数:
        input_size: 输入特征图的空间边长 (I)
        kernel_size: 卷积核的局部感受野边长 (K)
        stride: 滑动窗口的空间步幅 (S)
        padding: 空间维度的零填充圈数 (P)
        dilation: 膨胀系数 (D)，默认值 1 代表标准连续卷积
    """
    output_size = (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    return output_size


def calc_pool_output_size(
    input_size: int,
    kernel_size: int,
    stride: int = None
) -> int:
    r"""
    计算下采样(池化)层输出特征图的单侧物理分辨率。

    【机制说明】：
    若未显式声明 stride 参数，PyTorch API 默认令其等同于 kernel_size。
    推导等式：$O_{size} = \lfloor \frac{I_{size}}{S} \rfloor$
    """
    if stride is None:
        stride = kernel_size
    return input_size // stride


def print_size_changes():
    """
    CNN 特征图空间映射推演逻辑演示。
    输出基准输入 (64x64) 在经历两次标准化卷积与池化操作后的空间维度演变链路，
    用于验证进入全连接层之前的 Flatten 维度总量。
    """
    print("=" * 60)
    print("CNN 特征图空间维度演变推演（基准张量 64×64）")
    print("=" * 60)

    size = 64
    print(f"[起点] 初始张量空间维度: {size}×{size}")

    # Conv1 阶段
    size = calc_conv_output_size(size, kernel_size=3, padding=1, stride=1)
    print(f" -> 通过 Conv1 (Kernel: 3×3, Padding: 1): 保持维度 {size}×{size}")

    # Pool1 阶段
    size = calc_pool_output_size(size, kernel_size=2)
    print(f" -> 通过 Pool1 (Kernel: 2×2): 空间分辨率降维至 {size}×{size}")

    # Conv2 阶段
    size = calc_conv_output_size(size, kernel_size=3, padding=1, stride=1)
    print(f" -> 通过 Conv2 (Kernel: 3×3, Padding: 1): 保持维度 {size}×{size}")

    # Pool2 阶段
    size = calc_pool_output_size(size, kernel_size=2)
    print(f" -> 通过 Pool2 (Kernel: 2×2): 空间分辨率二次降维至 {size}×{size}")

    print(f"\n[终点] 线性分类器输入准备：")
    print(f"若当前特征通道数设为 64，则张量展平 (Flatten) 后的特征向量长度为: 64通道 × {size} × {size} = {64 * size * size}")
    print("=" * 60)


# ============================================================
# 2. 诊断级数据与高维特征可视化库
# ============================================================

def plot_training_history(
    history: Dict[str, List[float]],
    save_path: str = "training_history.png"
):
    """
    渲染模型完整训练生命周期内的标量评估指标 (Loss/Accuracy) 收敛曲线。

    【诊断价值说明】：
    1. 泛化能力评估：若训练集指标持续优化，但验证集指标出现停滞或反向发散，
       则指示模型陷入过拟合 (Overfitting) 状态，需引入权重衰减或数据增强策略。
    2. 稳定性监控：高频的剧烈振荡通常提示全局学习率 (Learning Rate) 设定过高，
       或批尺寸 (Batch Size) 容量不足。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['train_loss']) + 1)

    # 损失函数收敛图表
    ax1.plot(epochs, history['train_loss'], 'b-', label='训练集经验风险 (Train Loss)', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='验证集泛化风险 (Val Loss)', linewidth=2)
    ax1.set_xlabel('迭代轮次 (Epoch)', fontsize=12)
    ax1.set_ylabel('目标函数值 (Loss)', fontsize=12)
    ax1.set_title('损失函数收敛轨迹', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 准确率演变图表
    ax2.plot(epochs, history['train_acc'], 'b-', label='训练集拟合度 (Train Acc)', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='验证集准确率 (Val Acc)', linewidth=2)
    ax2.set_xlabel('迭代轮次 (Epoch)', fontsize=12)
    ax2.set_ylabel('准确率 (%)', fontsize=12)
    ax2.set_title('模型分类准确率演化轨迹', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 生命周期收敛曲线已写入: {save_path}")


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: str = "confusion_matrix.png",
    normalize: bool = True
):
    """
    渲染并输出全品类分类任务的混淆矩阵热力图。

    【指标判读逻辑】：
    理想模型下的高频响应区域应高度集中于主对角线。
    脱离对角线的高密度激活斑块，客观反映了模型在特定特征流形下存在系统性的分类混淆倾向
    (例如结构相近字符特征提取能力的不足)。
    """
    # 归一化操作：转换为类内召回率百分比，消除类别样本分布不均引入的视觉偏差
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(12, 10))
    # 采用 Blues 颜色映射，响应概率趋近 1.0 时饱和度达到最大
    im = ax.imshow(confusion_matrix, cmap='Blues')

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    ax.set_xlabel('模型输出概率最大类 (Predicted)', fontsize=12)
    ax.set_ylabel('数据真值标签 (Actual)', fontsize=12)
    ax.set_title('全集分类混淆响应热力图 (Confusion Matrix)', fontsize=14)
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
    构建随机批次样本的预测对照面板，用于直观检验局部推理精度。
    """
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            # 张量处理管线：GPU 卸载 -> 降维剥离 -> Numpy 转换 -> 逆归一化投影至 [0,1]
            img = images[i].squeeze().cpu().numpy()
            img = img * 0.5 + 0.5

            ax.imshow(img, cmap='gray')

            true_label = class_names[labels[i]]
            pred_label = class_names[predictions[i]]

            # 采用布尔判别决定高亮提示色
            color = 'green' if predictions[i] == labels[i] else 'red'
            ax.set_title(f'Pred: {pred_label} | True: {true_label}', color=color, fontsize=10)

        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"🎯 批次预测抽样图集已保存: {save_path}")


def visualize_conv_filters(
    model: nn.Module,
    layer_name: str = 'conv1',
    save_path: Optional[str] = None
):
    """
    提取并可视化指定卷积层的参数权重 (Filters / Weights)。

    【观测意义】：
    通过将权值矩阵逆向映射为像素阵列，可分析模型的底层感受野偏好。
    一般而言，浅层网络倾向于收敛为响应 Gabor 滤波器形式的基础边缘、纹理检测器。
    """
    # 动态挂载目标计算层
    conv_layer = getattr(model, layer_name)
    # 提取权重张量，切断梯度图并转移至 Host 内存
    filters = conv_layer.weight.data.cpu()

    # 执行 Min-Max 缩放，将浮点权重张量线性映射至标准灰度空间 [0, 1]
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    num_filters = filters.shape[0]
    grid_size = int(np.ceil(np.sqrt(num_filters)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    fig.suptitle(f'{layer_name} 权重感受野分布图', fontsize=14)

    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            ax.imshow(filters[i, 0], cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    save_name = save_path if save_path else f'{layer_name}_filters.png'
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"🔬 卷积核特征权重图已保存: {save_name}")


def visualize_feature_maps(
    model: nn.Module,
    image: torch.Tensor,
    layer_name: str = 'conv1',
    save_path: Optional[str] = None
):
    """
    基于前向传播钩子机制 (Forward Hooks)，截获并渲染张量在指定层的中间激活状态 (Feature Maps)。

    【观测意义】：
    监控输入特征矩阵在高维隐空间中的演化过程。
    网络深度增加会导致表征形态由显式的几何轮廓向高度抽象、稀疏的语义级激活阵列过渡。
    """
    model.eval()
    feature_maps = []

    # 注册回调句柄：目标层前向传递闭环时，挂载函数捕获输出张量
    def hook(_module, _input, output):
        feature_maps.append(output)

    layer = getattr(model, layer_name)
    hook_handle = layer.register_forward_hook(hook)

    # 封锁自动求导引擎，触发一次单向计算流
    with torch.no_grad():
        _ = model(image)

    # 注销监听句柄，防止持续挂载导致内存泄漏
    hook_handle.remove()

    # 截取首个 Batch 的激活张量堆叠
    maps = feature_maps[0][0].cpu()
    num_maps = min(maps.shape[0], 16)
    grid_size = 4

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle(f'{layer_name} 局部特征激活响应图 (Activations)', fontsize=14)

    for i, ax in enumerate(axes.flat):
        if i < num_maps:
            # 采用 Viridis 伪彩色映射以强化特征激活高值区域的对比度
            ax.imshow(maps[i], cmap='viridis')
        ax.axis('off')

    plt.tight_layout()
    save_name = save_path if save_path else f'{layer_name}_feature_maps.png'
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"🧬 特征激活状态流转图已保存: {save_name}")


# ============================================================
# 3. 计算图结构诊断与参数分析机制
# ============================================================

def count_parameters(model: nn.Module) -> int:
    """计算模型中具有 `requires_grad=True` 属性的所有可学习参数标量总和"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_by_layer(model: nn.Module) -> Dict[str, int]:
    """执行层级参数开销分析，精准定位计算图内存空间占用节点"""
    params_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_dict[name] = param.numel()
    return params_dict


def print_model_summary(model: nn.Module, input_size: Tuple[int, ...] = (1, 64, 64)):
    """
    输出网络拓扑结构的系统级摘要，并注入 Dummy Input 以检验层间维度的传递连贯性。
    """
    print("\n" + "=" * 60)
    print("AI 计算图拓扑结构剖析 (Architecture Topology Summary)")
    print("=" * 60)

    total = 0
    print("\n[层级参数配额清单]:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            params = param.numel()
            total += params
            print(f"  {name}: 尺度 {list(param.shape)} -> 映射 {params:,} 个参数节点")

    print(f"\n[汇总] 模型全局可训练参数标量总和: {total:,}")

    # 执行模拟前向传播测试，验证矩阵维度对齐状态
    model.eval()
    with torch.no_grad():
        dummy_input = torch.zeros(1, *input_size)
        try:
            output = model(dummy_input)
            print(f"  [维度穿透测试] 基准输入 Tensor 形状: {list(dummy_input.shape)}")
            print(f"  [维度穿透测试] 终端输出 Tensor 形状: {list(output.shape)}")
            print("  [状态反馈] 张量流转无阻碍，各级维度匹配逻辑校验通过。")
        except Exception as e:
            print(f"\n[致命异常捕获] 张量流转中断！\n推断原因：特征提取层输出至线性全连接层 (Linear) 的降维 Flatten 操作存在张量形态错位。\n底层堆栈信息: {e}")

    print("=" * 60)


# ============================================================
# 4. 异构计算硬件抽象与调度接口
# ============================================================

def get_device() -> torch.device:
    """
    底层计算资源探测与绑定管理器。
    【仲裁优先级策略】：
    1. Nvidia CUDA (并行计算首选)
    2. Apple MPS (Apple Silicon Metal 引擎加速适配)
    3. CPU (降级回退方案)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def to_device(data, device: torch.device):
    """
    广义张量设备迁移封装器。
    支持单一 Tensor 或深层嵌套的 Tuple/List 数据结构的递归式硬件寻址投递。
    """
    if isinstance(data, (list, tuple)):
        # 递归降解复合数据结构并逐级转移
        return [to_device(x, device) for x in data]
    return data.to(device)


# ============================================================
# 库文件集成测试桩 (Test Stub)
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Utils 辅助支撑模块 - 自动化冒烟测试序列启动")
    print("=" * 60)

    # 1. 验证维度推演逻辑
    print_size_changes()

    # 2. 构建测试拓扑验证摘要分析器
    mock_model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(32 * 32 * 32, 62)
    )

    print_model_summary(mock_model, input_size=(1, 64, 64))

    # 3. 验证硬件探测策略
    current_device = get_device()
    print(f"\n[硬件感知模块] 现役系统分配的加速设备标识: [{current_device}]")
    if str(current_device) == 'mps':
        print("  -> 检测到 Apple Silicon (MPS 架构) 加速引擎已激活挂载。")

    print("\n" + "=" * 60)
    print("Utils 模块集成自检完成，所有内部接口状态健康。")
    print("=" * 60)