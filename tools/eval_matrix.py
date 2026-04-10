"""
模型深度诊断工具：生成混淆矩阵与软肋报告
运行方式: python -m tools.eval_matrix --net resnet
"""
import os
import argparse
from datetime import datetime
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 解决 Mac 绘图时的中文乱码问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 导入核心包
from src import create_dataloaders, get_device, CLASS_IDX_TO_NAME, NUM_CLASSES
from src.model import SimpleCNN, DetailedCNN, ResNet
from src.utils import plot_confusion_matrix

def plot_top_weaknesses(cm: np.ndarray, class_names: list, save_path: str, top_k: int = 15):
    """提取混淆矩阵中的错误并绘制软肋排行榜"""
    errors = []
    # 遍历整个矩阵，寻找非对角线上的值（即判断错误的样本）
    for true_idx in range(NUM_CLASSES):
        for pred_idx in range(NUM_CLASSES):
            if true_idx != pred_idx and cm[true_idx, pred_idx] > 0:
                errors.append({
                    'true_label': class_names[true_idx],
                    'pred_label': class_names[pred_idx],
                    'count': cm[true_idx, pred_idx]
                })

    # 按照错误次数从高到低排序，取出前 top_k 名
    errors.sort(key=lambda x: x['count'], reverse=True)
    top_errors = errors[:top_k]

    if not top_errors:
        print("太棒了，模型没有任何错误，无法生成软肋图！")
        return

    # 准备绘图数据 (翻转列表以便在水平条形图中将最多的错误排在最上面)
    labels = [f"真实 '{e['true_label']}' 错认为 '{e['pred_label']}'" for e in top_errors][::-1]
    counts = [e['count'] for e in top_errors][::-1]

    # 绘制条形图
    plt.figure(figsize=(10, 8))
    bars = plt.barh(labels, counts, color='salmon')
    plt.xlabel('错判次数 (Count)', fontsize=12)
    plt.title(f'模型软肋 TOP-{top_k} 排行榜 (最易混淆的字符对)', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # 在条形图末尾标上具体数字
    for bar in bars:
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f"{int(bar.get_width())} 次",
                 va='center', ha='left', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="生成混淆矩阵诊断图")
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--net', type=str, default='detailed', choices=['simple', 'detailed', 'resnet'])
    parser.add_argument('--data_dir', type=str, default='data/processed')
    args = parser.parse_args()

    device = get_device()

    # ==========================================
    # 1. 动态生成档案夹结构
    # ==========================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"outputs/Diagnosis_{args.net.upper()}_{timestamp}"
    full_dir = os.path.join(report_dir, "1_full_matrix")
    weak_dir = os.path.join(report_dir, "2_weaknesses")

    os.makedirs(full_dir, exist_ok=True)
    os.makedirs(weak_dir, exist_ok=True)

    print("=" * 60)
    print(f"🚀 正在为 {args.net.upper()} 模型生成深度诊断报告...")
    print(f"📁 报告保存目录: {report_dir}")
    print("=" * 60)

    # ==========================================
    # 2. 挂载模型并准备数据
    # ==========================================
    if args.net == 'simple': model = SimpleCNN(num_classes=NUM_CLASSES)
    elif args.net == 'resnet': model = ResNet(num_classes=NUM_CLASSES)
    else: model = DetailedCNN(num_classes=NUM_CLASSES)

    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    model.eval()

    _, _, test_loader = create_dataloaders(data_dir=args.data_dir, batch_size=128, num_workers=0)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    # ==========================================
    # 3. 收集盲测数据
    # ==========================================
    print("\n🔍 正在对测试集进行盲测，并统计错题本...")
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="推理进度"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)

            # 填充混淆矩阵
            for t, p in zip(target.cpu().numpy(), predicted.cpu().numpy()):
                cm[t][p] += 1

    class_names = [CLASS_IDX_TO_NAME[i] for i in range(NUM_CLASSES)]

    # ==========================================
    # 4. 生成报告
    # ==========================================
    # 任务 A: 生成全景混淆矩阵
    full_matrix_path = os.path.join(full_dir, "confusion_matrix_full.png")
    plot_confusion_matrix(cm, class_names, save_path=full_matrix_path, normalize=True)
    print(f"✅ 全景混淆矩阵已保存至: {full_matrix_path}")

    # 任务 B: 提取并生成软肋排行榜图
    weakness_chart_path = os.path.join(weak_dir, "top_weaknesses_chart.png")
    plot_top_weaknesses(cm, class_names, save_path=weakness_chart_path, top_k=15)
    print(f"✅ 模型软肋排行榜已保存至: {weakness_chart_path}")

    print("\n" + "=" * 60)
    print("🎯 诊断全部完成！建议优先查看软肋排行榜进行针对性优化。")

if __name__ == "__main__":
    main()