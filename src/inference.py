"""
实时预测脚本 (推理引擎) - 完整修复版
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
# 使用非交互式后端
matplotlib.use('Agg')

# 导入本地模块
from .model import DetailedCNN, SimpleCNN, ResNet
from .dataset import CLASS_IDX_TO_NAME, CLASS_NAME_TO_IDX, NUM_CLASSES
from .utils import get_device

# 解决 Mac 版乱码
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 图像预处理与加载
# ============================================================

def get_inference_transform(image_size: int = 64) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def load_image(image_path: str, transform: transforms.Compose) -> torch.Tensor:
    image = Image.open(image_path)
    tensor = transform(image)
    return tensor.unsqueeze(0)

# ============================================================
# 预测器类 (推理引擎) - 【修复了嵌套错误】
# ============================================================

class Predictor:
    def __init__(self, model_path: str, net_type: str = 'detailed'):
        self.device = get_device()
        self.net_type = net_type
        self.class_names = CLASS_IDX_TO_NAME
        self.transform = get_inference_transform(64)
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str):
        # 核心：根据 net_type 匹配正确的锁芯
        if self.net_type == 'simple':
            model = SimpleCNN(num_classes=NUM_CLASSES)
        elif self.net_type == 'resnet':
            model = ResNet(num_classes=NUM_CLASSES)
        else:
            model = DetailedCNN(num_classes=NUM_CLASSES)

        model = model.to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到模型权重文件: {model_path}")

        # 加载权重
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def predict(self, image_input) -> dict:
        if isinstance(image_input, str):
            tensor = load_image(image_input, self.transform)
        elif isinstance(image_input, torch.Tensor):
            tensor = image_input
            if tensor.dim() == 3: tensor = tensor.unsqueeze(0)
        else:
            raise ValueError("不支持的输入类型")

        tensor = tensor.to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = probabilities.max(1)
            class_idx = predicted.item()

        return {
            'class_idx': class_idx,
            'class_name': CLASS_IDX_TO_NAME[class_idx],
            'confidence': confidence.item(),
            'probabilities': probabilities[0].cpu().numpy()
        }

    def predict_top_k(self, image_input, k: int = 5) -> List[Tuple[int, str, float]]:
        result = self.predict(image_input)
        probs = result['probabilities']
        top_k_indices = probs.argsort()[-k:][::-1]
        return [(idx, CLASS_IDX_TO_NAME[idx], probs[idx]) for idx in top_k_indices]

    def predict_batch(self, image_paths: List[str]) -> List[dict]:
        results = []
        for path in image_paths:
            try:
                res = self.predict(path)
                res['image_path'] = path
                results.append(res)
            except Exception as e:
                print(f"跳过异常图像 {path}: {e}")
        return results

# ============================================================
# 可视化工具
# ============================================================

def visualize_prediction(image_path: str, predictor: Predictor, save_path: str):
    image = Image.open(image_path)
    result = predictor.predict(image_path)
    top_k = predictor.predict_top_k(image_path, k=5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(image.convert('L'), cmap='gray')
    ax1.set_title(f"预测: {result['class_name']}\n({result['confidence']:.2%})")
    ax1.axis('off')

    names = [r[1] for r in top_k]
    probs = [r[2] for r in top_k]
    ax2.barh(names, probs, color=['green'] + ['steelblue']*4)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n📊 预测报告已生成: {save_path}")

def visualize_batch_predictions(image_paths: List[str], predictor: Predictor, save_path: str):
    n = min(len(image_paths), 16)
    rows = (n + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))

    for i, ax in enumerate(axes.flat):
        if i < n:
            img = Image.open(image_paths[i])
            res = predictor.predict(image_paths[i])
            ax.imshow(img.convert('L'), cmap='gray')
            ax.set_title(f"{res['class_name']} ({res['confidence']:.0%})",
                         color='green' if res['confidence']>0.8 else 'orange')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"🧩 批量画廊已生成: {save_path}")

# ============================================================
# 命令行入口 - 【修复了重复初始化问题】
# ============================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(description="CNN 字符识别推理终端")
    parser.add_argument('--model', '-m', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--image', '-i', type=str)
    parser.add_argument('--dir', '-d', type=str)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--net', type=str, default='detailed', choices=['simple', 'detailed', 'resnet'])
    parser.add_argument('--output', '-o', type=str, default=f'outputs/result_{timestamp}.png')

    args = parser.parse_args()

    print("=" * 60)
    print(f"CNN 推理终端 | 模式: {args.net.upper()} | 设备: {get_device()}")
    print("=" * 60)

    try:
        # 【关键】：只初始化一次，且必须传入正确的 net_type
        predictor = Predictor(args.model, net_type=args.net)
    except Exception as e:
        print(f"\n[致命错误]: {e}")
        return

    if args.image:
        if os.path.exists(args.image):
            res = predictor.predict(args.image)
            print(f"\n预测结果: {res['class_name']} ({res['confidence']:.2%})")
            visualize_prediction(args.image, predictor, args.output)

    elif args.dir:
        # 使用 rglob 递归搜索子文件夹
        image_paths = [str(p) for p in Path(args.dir).rglob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        if image_paths:
            print(f"总共找到 {len(image_paths)} 张图片，正在批量处理...")
            predictor.predict_batch(image_paths)
            visualize_batch_predictions(image_paths, predictor, args.output)
        else:
            print("未找到图片。")

    elif args.interactive:
        print("\n输入 'q' 退出交互模式")
        while True:
            path = input("\n请输入图片路径 >>> ").strip()
            if path.lower() in ['q', 'quit']: break
            if os.path.exists(path):
                res = predictor.predict(path)
                print(f"预测结果: '{res['class_name']}' ({res['confidence']:.2%})")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()