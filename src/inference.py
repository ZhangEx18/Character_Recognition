"""
实时预测脚本 (推理引擎) - 完整交付版

【系统架构定位】
本模块是整个 CNN 系统的“输出端”与“实战装甲”。它将训练好的静止权重文件 (.pth)
唤醒为可以实时处理图像的智能大脑。

【核心特性】
1. 架构兼容：动态支持 SimpleCNN, DetailedCNN, ResNet 三种网络架构的无缝切换。
2. 面向对象：高度封装的 Predictor 类，可作为黑盒被 FastAPI/Flask 或 GUI 程序直接调用。
3. 纯净环境：与 dataset.py 完全解耦，推理时不需要原训练集存在，支持独立物理部署。
4. 容错流水线：批量预测时自带异常捕获，单张图片损坏不会导致整个测试任务崩溃。
"""

import os
import argparse
from pathlib import Path
from typing import Tuple, List, Optional
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib

# 【工程细节】：强制设定 Matplotlib 的后端为 'Agg'
# 作用：在没有连接显示器（如远程 Linux 服务器、云主机）的环境下画图时，
# 如果不加这行，程序会因为找不到 GUI 界面而直接崩溃报错。
matplotlib.use('Agg')

# 导入本地项目核心组件
from .model import DetailedCNN, SimpleCNN, ResNet
from .dataset import CLASS_IDX_TO_NAME, CLASS_NAME_TO_IDX, NUM_CLASSES
from .utils import get_device

# 解决 Mac 环境下 Matplotlib 渲染中文标签时出现的“豆腐块”乱码问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 图像预处理与加载流水线 (Inference Pipeline)
# ============================================================

def get_inference_transform(image_size: int = 64) -> transforms.Compose:
    """
    获取推理阶段专用的图像预处理流水线。

    【核心区别】：
    与训练阶段不同，推理阶段【绝对不能】使用 RandomRotation(随机旋转) 等数据增强！
    在考场上，我们必须把考卷（图片）原汁原味、清晰端正地递给模型。
    """
    return transforms.Compose([
        # 1. 物理缩放：强制对齐模型输入层尺寸
        transforms.Resize((image_size, image_size)),
        # 2. 灰度化：强制转换为单通道。防止用户传入 RGB 彩色图导致模型通道报错
        transforms.Grayscale(num_output_channels=1),
        # 3. 张量化：将像素值从 [0, 255] 压缩至 [0.0, 1.0]，并转换为 PyTorch 张量
        transforms.ToTensor(),
        # 4. 数学归一化：将 [0.0, 1.0] 映射到 [-1.0, 1.0]，必须与训练时严格保持一致！
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def load_image(image_path: str, transform: transforms.Compose) -> torch.Tensor:
    """
    从物理硬盘加载图像，并清洗为模型可吞吐的标准张量。
    """
    image = Image.open(image_path)
    tensor = transform(image)

    # 【高频踩坑点】：PyTorch 模型的输入维度要求是 (Batch_Size, Channels, Height, Width)。
    # 哪怕只预测 1 张图，也不能直接传 (1, 64, 64)，必须补齐 Batch 维度变成 (1, 1, 64, 64)。
    # unsqueeze(0) 就是在第 0 个位置强行插入一个维度。
    return tensor.unsqueeze(0)


# ============================================================
# 核心预测器类 (面向对象封装的黑盒引擎)
# ============================================================

class Predictor:
    """
    字符识别预测器。
    它负责隐藏底层复杂的模型加载、设备流转 (CPU/GPU/MPS) 和数学计算。
    """
    def __init__(self, model_path: str, net_type: str = 'detailed'):
        self.device = get_device()
        self.net_type = net_type
        self.class_names = CLASS_IDX_TO_NAME
        self.transform = get_inference_transform(64)

        # 初始化时立即执行模型解析与权重挂载
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> nn.Module:
        """根据指定的网络类型，实例化正确的锁芯并插入权重钥匙"""

        # 1. 动态路由：根据 net_type 实例化空壳模型
        if self.net_type == 'simple':
            model = SimpleCNN(num_classes=NUM_CLASSES)
        elif self.net_type == 'resnet':
            model = ResNet(num_classes=NUM_CLASSES)
        else:
            model = DetailedCNN(num_classes=NUM_CLASSES)

        # 2. 将空壳模型发送至计算硬件 (Mac MPS 或 GPU)
        model = model.to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到模型权重文件: {model_path}")

        # 3. 加载权重字典，并映射到当前可用设备
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)

        # 4. 【关键操作】：切换为评估模式。
        # 这会关闭 Dropout 的随机丢弃，并冻结 BatchNorm 的均值和方差更新。
        model.eval()
        return model

    def predict(self, image_input) -> dict:
        """
        全能单图预测接口。
        支持传入文件路径 (str)，或者已经在内存中的张量 (Tensor)。
        """
        # --- 数据清洗与格式化 ---
        if isinstance(image_input, str):
            tensor = load_image(image_input, self.transform)
        elif isinstance(image_input, torch.Tensor):
            tensor = image_input
            if tensor.dim() == 3: tensor = tensor.unsqueeze(0)
        else:
            raise ValueError("不支持的输入类型，仅支持路径字符串或张量。")

        tensor = tensor.to(self.device)

        # --- 前向计算 ---
        # 【黄金法则】：torch.no_grad() 会彻底关闭梯度计算引擎。
        # 在推理时开启它，可以节省 50% 以上的显存，并大幅提升计算速度。
        with torch.no_grad():
            output = self.model(tensor)

            # 模型输出的是没有边界的 Logits。
            # 必须用 Softmax 函数将它们压缩成总和为 1 的“概率分布”。
            probabilities = F.softmax(output, dim=1)

            # 提取概率最高的值 (置信度) 和对应的索引 (预测结果)
            confidence, predicted = probabilities.max(1)
            class_idx = predicted.item()

        return {
            'class_idx': class_idx,
            'class_name': CLASS_IDX_TO_NAME[class_idx],
            'confidence': confidence.item(),
            'probabilities': probabilities[0].cpu().numpy() # 剥离出计算图，转为 NumPy 数组供后续使用
        }

    def predict_top_k(self, image_input, k: int = 5) -> List[Tuple[int, str, float]]:
        """
        获取概率最高的前 K 个候选结果。
        业务场景：当第一名置信度不高时（比如 40%），展示第 2、3 名可以辅助人类做最终判断。
        """
        result = self.predict(image_input)
        probs = result['probabilities']

        # argsort() 会将数组从小到大排序并返回索引。
        # [::-1] 将其翻转为从大到小，再截取前 k 个（即概率最大的 k 个）。
        if k <= 0:
            return []
        top_k_indices = probs.argsort()[::-1][:k]

        return [(idx, CLASS_IDX_TO_NAME[idx], probs[idx]) for idx in top_k_indices]

    def predict_batch(self, image_paths: List[str]) -> List[dict]:
        """
        鲁棒性极强的批量预测函数。
        如果包含损坏的图片，它会打印警告并跳过，绝不让整个批次崩溃。
        """
        results = []
        for path in image_paths:
            try:
                res = self.predict(path)
                res['image_path'] = path
                results.append(res)
            except Exception as e:
                print(f"跳过异常或损坏的图像 {path}: {e}")
        return results


# ============================================================
# 数据可视化表现层 (Visualization Layer)
# ============================================================

def visualize_prediction(image_path: str, predictor: Predictor, save_path: str):
    """单张图片的深度体检报告：左侧显示原图，右侧显示 Top-5 概率柱状图"""
    image = Image.open(image_path)
    result = predictor.predict(image_path)
    top_k = predictor.predict_top_k(image_path, k=5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # L 模式保证纯粹的灰度显示，防止受原图奇怪色彩通道的干扰
    ax1.imshow(image.convert('L'), cmap='gray')
    ax1.set_title(f"预测: {result['class_name']}\n({result['confidence']:.2%})")
    ax1.axis('off')

    names = [r[1] for r in top_k]
    probs = [r[2] for r in top_k]

    # 视觉小巧思：第一名标绿（冠军），剩下的陪跑者标蓝
    ax2.barh(names, probs, color=['green'] + ['steelblue']*4)
    ax2.invert_yaxis() # 翻转 Y 轴，让冠军排在最上面
    ax2.set_xlim(0, 1) # 概率的满分固定为 1 (即 100%)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n📊 预测体检报告已生成: {save_path}")

def visualize_batch_predictions(image_paths: List[str], predictor: Predictor, save_path: str):
    """批量预测画廊：生成 4x4 的图片墙，快速审阅批量结果"""
    n = min(len(image_paths), 16) # 防爆显存：最多只画 16 张图
    rows = (n + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))

    for i, ax in enumerate(axes.flat):
        if i < n:
            img = Image.open(image_paths[i])
            res = predictor.predict(image_paths[i])
            ax.imshow(img.convert('L'), cmap='gray')

            # 视觉红绿灯：>80% 标绿，信心不足标橙色警示
            ax.set_title(f"{res['class_name']} ({res['confidence']:.0%})",
                         color='green' if res['confidence']>0.8 else 'orange')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"🧩 批量画廊已生成: {save_path}")


# ============================================================
# 命令行终端 (CLI) 调度入口
# ============================================================

def main():
    # 生成带时间戳的默认输出文件名，防止之前辛苦跑出来的测试图被瞬间覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(description="CNN 字符识别终极推理终端")
    parser.add_argument('--model', '-m', type=str, default='checkpoints/best_model.pth', help='模型权重文件所在路径')
    parser.add_argument('--image', '-i', type=str, help='单图诊断模式：输入一张图片的绝对或相对路径')
    parser.add_argument('--dir', '-d', type=str, help='文件夹扫荡模式：输入一个文件夹路径，自动提取图片并批量预测')
    parser.add_argument('--interactive', action='store_true', help='互动模式：开启终端文字聊天式的连续推断')
    parser.add_argument('--net', type=str, default='detailed', choices=['simple', 'detailed', 'resnet'], help='指定模型大脑的架构类型')
    parser.add_argument('--output', '-o', type=str, default=f'outputs/result_{timestamp}.png', help='生成的分析图表保存位置')

    args = parser.parse_args()

    print("=" * 60)
    print(f"CNN 推理终端 | 核心引擎: {args.net.upper()} | 加速设备: {get_device()}")
    print("=" * 60)

    try:
        # 【生命周期】：无论选了什么模式，Predictor 都只在这里初始化一次！
        # 彻底避免了之前在内存中反复加载模型导致的内存泄漏和报错。
        predictor = Predictor(args.model, net_type=args.net)
    except Exception as e:
        print(f"\n[致命错误]: {e}")
        return

    # --- 模式 A: 单图体检 ---
    if args.image:
        if os.path.exists(args.image):
            res = predictor.predict(args.image)
            print(f"\n预测结果: {res['class_name']} ({res['confidence']:.2%})")
            visualize_prediction(args.image, predictor, args.output)

    # --- 模式 B: 目录批量扫荡 ---
    elif args.dir:
        # rglob('*') 是神级操作：它会【递归】穿透进入所有的子文件夹去寻找图片。
        # 完美适配 data/processed/test 这种按类别分子文件夹的数据集结构。
        image_paths = [str(p) for p in Path(args.dir).rglob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        if image_paths:
            print(f"总共找到 {len(image_paths)} 张图片，正在执行矩阵推断...")
            predictor.predict_batch(image_paths)
            visualize_batch_predictions(image_paths, predictor, args.output)
        else:
            print("未能在此目录层级下找到任何支持的图片格式。")

    # --- 模式 C: 终端文字交互 ---
    elif args.interactive:
        print("\n已进入交互模式 (输入 'q' 退出当前会话)")
        while True:
            path = input("\n[终端输入] 图片物理路径 >>> ").strip()
            if path.lower() in ['q', 'quit']: break
            if os.path.exists(path):
                res = predictor.predict(path)
                print(f"[AI 判定]: 这是字符 '{res['class_name']}' (置信度: {res['confidence']:.2%})")
            else:
                print("路径似乎有误，请重新输入。")

    else:
        # 如果什么参数都不传，弹出版面帮助
        parser.print_help()

if __name__ == "__main__":
    main()