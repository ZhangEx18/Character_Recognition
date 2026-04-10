"""
实时预测脚本 (推理引擎)

功能：
1. 加载训练好的模型 (支持从断点或纯权重加载)
2. 单张图片预测 (输出最可能的结果与 Top-5)
3. 批量预测 (遍历文件夹快速推断)
4. 预测结果可视化 (生成概率条形图对比)
5. 纯净环境运行 (完全不依赖原训练集数据，可独立部署)

使用方法：
    # 预测单张图片
    python -m src.inference --image path/to/image.png

    # 批量预测目录下的所有图片
    python -m src.inference --dir path/to/images/

    # 交互式预测 (循环输入路径)
    python -m src.inference --interactive
"""

import os
import argparse
from pathlib import Path
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
# 使用非交互式后端，确保在无界面的服务器上也能正常生成并保存图片，不报 GUI 错误
matplotlib.use('Agg')

# 导入本地模块 (使用相对导入)
from .model import DetailedCNN, SimpleCNN
from .dataset import CLASS_IDX_TO_NAME, CLASS_NAME_TO_IDX, NUM_CLASSES
from .utils import get_device
import matplotlib.pyplot as plt

# 解决 Mac 版 PyCharm 运行结果中的中文乱码
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# ============================================================
# 图像预处理 (推理专用)
# ============================================================

def get_inference_transform(image_size: int = 64) -> transforms.Compose:
    """
    获取推理时的图像预处理流水线

    【核心区别】推理阶段绝对不能使用 RandomRotation 等数据增强手段！
    模型需要看到最原本的图像以给出最确定的判断。

    参数:
        image_size: 图像目标尺寸 (默认修改为 64 以匹配 DetailedCNN)

    返回:
        transforms.Compose 变换管道
    """
    return transforms.Compose([
        # 1. 缩放尺寸
        transforms.Resize((image_size, image_size)),
        # 2. 强制转为单通道灰度图，防止传入 RGB 图片导致通道数报错
        transforms.Grayscale(num_output_channels=1),
        # 3. 转为 PyTorch 张量，并将像素值从 [0, 255] 自动缩放到 [0.0, 1.0]
        transforms.ToTensor(),
        # 4. 归一化：(x - 0.5) / 0.5，将 [0.0, 1.0] 映射到 [-1.0, 1.0]，与训练时对齐
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def load_image(image_path: str, transform: transforms.Compose) -> torch.Tensor:
    """
    加载并处理来自硬盘的图片，使其成为模型可接受的张量

    参数:
        image_path: 图像物理路径
        transform: 预处理管道函数

    返回:
        预处理后的张量，形状为 (1, 1, 64, 64) -> (batch_size, channels, H, W)
    """
    image = Image.open(image_path)
    tensor = transform(image)

    # 模型要求输入必须有 batch_size 维度，即使只有一张图
    # unsqueeze(0) 会在第 0 维插入一个维度：(1, 64, 64) -> (1, 1, 64, 64)
    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# 预测器类 (推理引擎)
# ============================================================

class Predictor:
    """
    字符识别预测器 (面向对象封装)

    将繁琐的模型加载、设备分配、预处理和后处理封装成黑盒。
    便于被其他 Web 框架（如 Flask/FastAPI）或 GUI 程序直接调用。

    使用示例:
        predictor = Predictor("checkpoints/best_model.pth")
        result = predictor.predict("test_image.png")
        print(f"识别结果: {result['class_name']}")
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        image_size: int = 64
    ):
        """初始化预测器环境"""
        # 如果未指定设备，自动检测 GPU/MPS/CPU
        self.device = device or get_device()
        print(f"初始化预测器 | 使用设备: {self.device}")

        self.image_size = image_size
        # 提前初始化好转换管道，避免每次预测时重复创建
        self.transform = get_inference_transform(image_size)

        # 挂载模型
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> nn.Module:
        """内部方法：安全加载模型权重"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型权重文件不存在，请检查路径: {model_path}")

        # 实例化网络结构 (此处写死为性能更好的 DetailedCNN，需确保输入尺寸为 64x64)
        model = DetailedCNN(num_classes=NUM_CLASSES)

        # map_location 极为重要：它允许我们在没有 GPU 的机器上加载用 GPU 训练出的模型
        checkpoint = torch.load(model_path, map_location=self.device)

        # 兼容性处理：支持纯权重文件和带有训练状态的 Checkpoint 文件
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("成功从完整 Checkpoint 中提取纯净权重。")
        else:
            state_dict = checkpoint

        # 将权重注入模型架构
        model.load_state_dict(state_dict)
        model = model.to(self.device)

        # 【关键】切换到评估模式：关闭 Dropout，锁定 BatchNorm 的均值和方差
        model.eval()

        return model

    def predict(self, image_input) -> dict:
        """
        核心推理方法：预测单张图像

        参数:
            image_input: 可以是物理文件路径 (str)，也可以是内存中的预处理张量 (Tensor)

        返回:
            包含详细预测指标的字典
        """
        # 1. 统一输入类型格式
        if isinstance(image_input, str):
            tensor = load_image(image_input, self.transform)
        elif isinstance(image_input, torch.Tensor):
            tensor = image_input
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
        else:
            raise ValueError("不支持的输入类型，请传入路径字符串或 PyTorch 张量")

        tensor = tensor.to(self.device)

        # 2. 闭包前向传播
        # torch.no_grad() 是推理时的黄金法则：它告诉 PyTorch 不要记录计算图，节约大量内存并加速运算
        with torch.no_grad():
            output = self.model(tensor)

            # 模型输出的是无界的 raw logits。需使用 Softmax 转换成总和为 1 的概率分布
            probabilities = F.softmax(output, dim=1)

            # 获取概率最大值 (confidence置信度) 及其对应的索引 (predicted预测类别)
            confidence, predicted = probabilities.max(1)

            class_idx = predicted.item()
            confidence = confidence.item()

        # 3. 组装结果字典
        return {
            'class_idx': class_idx,
            'class_name': CLASS_IDX_TO_NAME[class_idx],
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy()  # 返回所有 62 个类别的概率数组
        }

    def predict_batch(self, image_paths: List[str]) -> List[dict]:
        """批量预测工具函数，封装了异常处理，防止单张图片损坏导致整个批次崩溃"""
        results = []
        for path in image_paths:
            try:
                result = self.predict(path)
                result['image_path'] = path
                results.append(result)
            except Exception as e:
                print(f"跳过损毁或异常的图像 {path} | 错误信息: {e}")
        return results

    def predict_top_k(self, image_input, k: int = 5) -> List[Tuple[int, str, float]]:
        """
        获取 Top-K 候选结果
        在很多场景下，第一名置信度不高时，参考前 K 名有助于辅助人工决策。
        """
        result = self.predict(image_input)
        probs = result['probabilities']

        # argsort() 返回的是从小到大排序的索引
        # [-k:] 取出最后 k 个（即最大的 k 个），[::-1] 将其翻转为从大到小
        top_k_indices = probs.argsort()[-k:][::-1]

        top_k_results = []
        for idx in top_k_indices:
            top_k_results.append((
                idx,
                CLASS_IDX_TO_NAME[idx],
                probs[idx]
            ))

        return top_k_results


# ============================================================
# 结果可视化层
# ============================================================

def visualize_prediction(
    image_path: str,
    predictor: Predictor,
    save_path: Optional[str] = None
):
    """生成并保存单张图像的详细预测报告图（含 Top-5 概率柱状图）"""
    image = Image.open(image_path)

    result = predictor.predict(image_path)
    top_k = predictor.predict_top_k(image_path, k=5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # 左侧：显示原图
    # convert('L') 确保它以单通道灰度渲染，防止 matplotlib 颜色映射异常
    ax1.imshow(image.convert('L'), cmap='gray')
    ax1.set_title(f"预测判定: {result['class_name']}\n(置信度: {result['confidence']:.2%})")
    ax1.axis('off')

    # 右侧：水平柱状图显示前 5 名候选
    names = [r[1] for r in top_k]
    probs = [r[2] for r in top_k]

    # 最高概率标绿，其余备选标蓝
    colors = ['green' if i == 0 else 'steelblue' for i in range(len(names))]
    ax2.barh(names, probs, color=colors)
    ax2.set_xlabel('概率 (Probability)')
    ax2.set_title('Top-5 候选预测分布')
    ax2.set_xlim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 单张预测可视化报告已生成: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_batch_predictions(
    image_paths: List[str],
    predictor: Predictor,
    save_path: str = "batch_predictions.png"
):
    """将多张图像的预测结果拼接到一张大网格图中进行预览"""
    n = min(len(image_paths), 16) # 最多展示 16 张，防止图片过大导致内存溢出
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))

    for i, ax in enumerate(axes.flat):
        if i < n:
            image = Image.open(image_paths[i])
            result = predictor.predict(image_paths[i])

            ax.imshow(image.convert('L'), cmap='gray')
            ax.set_title(f"{result['class_name']} ({result['confidence']:.0%})",
                         color='green' if result['confidence'] > 0.8 else 'orange')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n🧩 批量预测画廊缩略图已生成: {save_path}")


# ============================================================
# 命令行终端接口
# ============================================================

def main():
    """解析命令行参数并派发执行逻辑"""
    parser = argparse.ArgumentParser(description='CNN 字符识别 - 推理部署模块')

    # 1. 修正模型路径参数：必须有 --model 且缩写为 -m
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='checkpoints/best_model.pth',
        help='编译好的模型权重物理路径'
    )

    # 2. 输入相关参数
    parser.add_argument('--image', '-i', type=str, help='指定进行单张测试的图片路径')
    parser.add_argument('--dir', '-d', type=str, help='指定需要批量测试的图片文件夹路径')
    parser.add_argument('--interactive', action='store_true', help='开启终端交互式问答预测模式')

    # 3. 修正输出参数：全文件只保留这一个 --output / -o
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs/prediction_result.png',
        help='生成的分析图保存路径'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CNN 字符识别推理终端")
    print("=" * 60)

    # 尝试拉起推理引擎
    try:
        # 此时 args.model 才能正确生效
        predictor = Predictor(args.model)
    except FileNotFoundError as e:
        print(f"\n[致命错误]: {e}")
        print("您需要先运行 train.py 训练网络，或确保 --model 传入了正确的 .pth 文件路径。")
        return

    # --- 后续逻辑（单图、批量、交互）保持不变 ---
    if args.image:
        if not os.path.exists(args.image):
            print(f"找不到指定的测试图片: {args.image}")
            return
        result = predictor.predict(args.image)
        print(f"\n预测字符: {result['class_name']} (置信度: {result['confidence']:.2%})")
        visualize_prediction(args.image, predictor, args.output)

    elif args.dir:
        image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_paths.extend(Path(args.dir).glob(ext))
        if not image_paths:
            print("未能在此文件夹中找到支持的图片格式。")
            return
        predictor.predict_batch([str(p) for p in image_paths])
        visualize_batch_predictions([str(p) for p in image_paths[:16]], predictor, args.output)

    elif args.interactive:
        print("\n进入交互式模式...")
        while True:
            path = input("\n[输入] 图像路径 >>> ").strip()
            if path.lower() in ['q', 'quit']: break
            if not os.path.exists(path): continue
            res = predictor.predict(path)
            print(f"[输出] 字符: '{res['class_name']}' ({res['confidence']:.2%})")
    else:
        parser.print_help()

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()