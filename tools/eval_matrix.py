"""
模型深度诊断工具：生成混淆矩阵与软肋报告 (Model Diagnostic & Profiling Tool)

【系统架构定位】
本脚本是模型的“期末考试阅卷机”与“体检报告生成器”。
它不会改变模型的权重，而是用客观的测试集数据去全面探查模型在 62 类字符中的真实表现。
通过计算，它能精准定位出模型“最容易搞混的字符双胞胎”（如 0和O，1和l），
为您下一步针对性地做数据增强或结构优化指明方向。

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

# 【工程避坑】：强制使用无图形界面的 'Agg' 渲染后端。
# 如果不加这行，在纯终端或无显示器的服务器上画图时，底层库会试图弹窗而导致程序崩溃报错。
matplotlib.use('Agg')

# 解决 Mac 系统下 Matplotlib 绘图时中文显示为方块（豆腐块）的乱码问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 从核心层导入我们需要的数据工厂、硬件管家和工具集
from src import create_dataloaders, get_device, CLASS_IDX_TO_NAME, NUM_CLASSES
from src.model import SimpleCNN, DetailedCNN, ResNet
from src.utils import plot_confusion_matrix


# ============================================================
# 核心诊断逻辑：提取错题并生成“软肋排行榜”
# ============================================================
def plot_top_weaknesses(cm: np.ndarray, class_names: list, save_path: str, top_k: int = 15):
    """
    数据挖掘函数：像淘金一样从 62x62 的巨大混淆矩阵中，过滤出错误率最高的重灾区，并画成条形图。

    参数:
        cm: 混淆矩阵 (二维的 Numpy 数组，记录了所有的判定次数)
        class_names: 62 个类别的名字列表
        save_path: 图片保存路径
        top_k: 只展示排名前 K 的致命错误 (默认前 15 名)
    """
    errors = []

    # 1. 遍历整个 62x62 的矩阵网络
    # true_idx 代表行（这道题的标准答案）
    for true_idx in range(NUM_CLASSES):
        # pred_idx 代表列（AI 考生给出的答案）
        for pred_idx in range(NUM_CLASSES):
            # 【核心过滤逻辑】：
            # 我们只关心 true_idx != pred_idx 的格子（因为横纵坐标相等代表猜对了）。
            # 并且错误次数必须大于 0，否则没必要记录。
            if true_idx != pred_idx and cm[true_idx, pred_idx] > 0:
                errors.append({
                    'true_label': class_names[true_idx],   # 真实身份
                    'pred_label': class_names[pred_idx],   # 委屈的误判身份
                    'count': cm[true_idx, pred_idx]        # 到底错认了多少次
                })

    # 2. 排序：根据错误次数 ('count') 进行降序排列 (reverse=True)
    errors.sort(key=lambda x: x['count'], reverse=True)

    # 3. 截断：我们不关心那些只错了一两次的偶然事件，只抓取前 top_k 个“顽固错误”
    top_errors = errors[:top_k]

    if not top_errors:
        print("🎉 太棒了，矩阵对角线外全为 0，模型没有任何错误，无法生成软肋图！")
        return

    # 4. 准备绘图数据
    # 【UI 细节】：因为我们用的是水平条形图 (barh)，Matplotlib 画图是从下往上画的。
    # 为了让错误次数最多的排在图表最顶端（最醒目），我们使用 [::-1] 将列表倒序。
    labels = [f"真实 '{e['true_label']}' 错认为 '{e['pred_label']}'" for e in top_errors][::-1]
    counts = [e['count'] for e in top_errors][::-1]

    # 5. 渲染图表
    plt.figure(figsize=(10, 8))
    # 使用 'salmon' (三文鱼粉红) 这种偏红的警示色来凸显“错误”的概念
    bars = plt.barh(labels, counts, color='salmon')
    plt.xlabel('错判次数 (Count)', fontsize=12)
    plt.title(f'模型软肋 TOP-{top_k} 排行榜 (最易混淆的字符对)', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7) # 加上辅助竖线，方便对齐看数量

    # 6. 数据刻度增强：直接在柱状图的尾巴上写出具体错了几次，免去用户向下对齐坐标轴的麻烦
    for bar in bars:
        plt.text(
            bar.get_width() + 0.1,             # X坐标：放在柱子末端的右边一点点
            bar.get_y() + bar.get_height()/2,  # Y坐标：对齐到柱子的垂直中心
            f"{int(bar.get_width())} 次",      # 打印具体数字
            va='center', ha='left', fontsize=10
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# 主调度函数：控制整个诊断流水线
# ============================================================
def main():
    # --- A. 命令行交互声明 ---
    parser = argparse.ArgumentParser(description="生成混淆矩阵深度诊断报告")
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth', help='模型权重的物理路径')
    parser.add_argument('--net', type=str, default='detailed', choices=['simple', 'detailed', 'resnet'], help='要诊断的网络架构')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='处理好的数据集根目录')
    args = parser.parse_args()

    # 自动获取最佳算力硬件 (MPS / CUDA / CPU)
    device = get_device()

    # ==========================================
    # 1. 动态生成档案夹结构 (版本控制机制)
    # ==========================================
    # 【工程思维】：做实验最怕的就是数据互相覆盖。
    # 我们用精确到秒的时间戳作为文件夹名，这样每次运行诊断都会留下独立的“病历档案”。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"outputs/Diagnosis_{args.net.upper()}_{timestamp}"
    full_dir = os.path.join(report_dir, "1_full_matrix")  # 子文件夹 1：放宏观全景图
    weak_dir = os.path.join(report_dir, "2_weaknesses")   # 子文件夹 2：放具体的错题排行榜

    os.makedirs(full_dir, exist_ok=True)
    os.makedirs(weak_dir, exist_ok=True)

    print("=" * 60)
    print(f"🚀 正在为 {args.net.upper()} 模型生成深度诊断报告...")
    print(f"📁 报告独立档案室: {report_dir}")
    print("=" * 60)

    # ==========================================
    # 2. 挂载模型并准备“阅卷”数据
    # ==========================================
    # 根据传入的参数，动态分配“锁芯”
    if args.net == 'simple': model = SimpleCNN(num_classes=NUM_CLASSES)
    elif args.net == 'resnet': model = ResNet(num_classes=NUM_CLASSES)
    else: model = DetailedCNN(num_classes=NUM_CLASSES)

    # 插入“钥匙”（加载权重）并推送到加速设备
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)

    # 【致命细节】：开启评估模式！这会关掉网络中的 Dropout 层。
    # 考试时绝不能蒙住 AI 的眼睛，必须让所有神经元 100% 发挥实力。
    model.eval()

    # 调用数据工厂，我们只需要测试集 (test_loader) 的数据，因为那是它没见过的“盲测试卷”
    _, _, test_loader = create_dataloaders(data_dir=args.data_dir, batch_size=128, num_workers=0)

    # 初始化一个 62x62 的全 0 矩阵，就像一张干净的计票纸，用来统计预测结果
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    # ==========================================
    # 3. 收集盲测数据 (推理主循环)
    # ==========================================
    print("\n🔍 正在对测试集进行全面盲测，并逐行核实错题本...")

    # 【黄金法则】：推理时必须包裹在 torch.no_grad() 中。
    # 关闭自动求导机制，可以直接砍掉一半的显存占用，并大幅提升推理速度。
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="阅卷进度"):
            # 将试卷数据搬运到 GPU/MPS 显存
            data, target = data.to(device), target.to(device)

            # 模型作答：给出 62 个类别的概率打分 (Logits)
            output = model(data)

            # 提取概率最高那一项的索引，作为最终答案
            _, predicted = output.max(1)

            # 遍历当前批次 (Batch) 中的每一张图
            # zip 让我们能同时拿到“正确答案 t”和“AI 的答案 p”
            for t, p in zip(target.cpu().numpy(), predicted.cpu().numpy()):
                # 在计票纸的对应格子上画“正”字加一
                # 如果猜对了，就会加在 (t, t) 也就是对角线上；猜错了，就会落在别的地方。
                cm[t][p] += 1

    # 提取所有 62 个类别的名字清单 (用于画图时的坐标轴刻度)
    class_names = [CLASS_IDX_TO_NAME[i] for i in range(NUM_CLASSES)]

    # ==========================================
    # 4. 生成可视化报告
    # ==========================================

    # 任务 A: 绘制那张 62x62 的巨大蓝色热力图 (全景混淆矩阵)
    full_matrix_path = os.path.join(full_dir, "confusion_matrix_full.png")
    # normalize=True 会在画图前把具体的错题数量转换成该类别的百分比，解决由于数据分布不均导致的颜色失准
    plot_confusion_matrix(cm, class_names, save_path=full_matrix_path, normalize=True)
    print(f"✅ 全景混淆矩阵已生成至: {full_matrix_path}")

    # 任务 B: 提取数据，生成粉红色的重点软肋排行榜
    weakness_chart_path = os.path.join(weak_dir, "top_weaknesses_chart.png")
    plot_top_weaknesses(cm, class_names, save_path=weakness_chart_path, top_k=15)
    print(f"✅ 模型软肋排行榜已生成至: {weakness_chart_path}")

    print("\n" + "=" * 60)
    print("🎯 诊断全部完成！建议优先查看【软肋排行榜】进行针对性的数据清洗或模型优化。")
    print("=" * 60)

if __name__ == "__main__":
    main()