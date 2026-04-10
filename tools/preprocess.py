"""
数据预处理与切分脚本 (Data Preprocessing & Splitter Pipeline)

【系统架构定位】
本脚本位于系统流水线的最前端（数据准备层）。
它负责将原始混乱、庞大、未分类的数据源，清洗并转换为标准化、轻量级的模型燃料，
并严格按照工业级标准，在物理硬盘层面上将数据强制划分为训练集(Train)、验证集(Val)和测试集(Test)。

【核心工程特性】
1. 物理隔离：直接在硬盘上建立 train/val/test 文件夹。这比用代码在内存中切分更安全，彻底杜绝数据泄露。
2. 增量处理：自带缓存记忆能力。如果中途断电或者新增了部分图片，再次运行只会处理新图，不会全部重头再来。
3. 极限瘦身：将彩色原图转为单通道灰度图并压缩尺寸，极大降低后续训练时的 I/O 和显存压力。

运行方式: 确保终端当前路径为【项目根目录】，运行: python -m tools.preprocess
"""

import os
import cv2
import random
from tqdm import tqdm

# ============================================================
# 全局配置中心 (Configuration Area)
# ============================================================

# 原始数据池的绝对或相对路径（存放所有按类别分好文件夹的原图）
RAW_DATA_DIR = 'data/raw/archive/augmented_images/augmented_images1'

# 清洗后标准数据的输出阵地
OUTPUT_DIR = 'data/processed'

# 物理裁剪目标尺寸。64x64 是兼顾了特征保留与计算性能的黄金尺寸。
TARGET_SIZE = (64, 64)

# 数据集切分比例字典 (Train : Val : Test)
# 【学术标准】：
# - 85% Train: 给模型当课本，用来学习。
# - 10% Val: 给模型当模拟考，用来动态调整学习率和防止过拟合。
# - 5% Test: 高考卷，在模型彻底训练完毕前绝对不能碰，用于最终的实力盲测。
# （注：请确保三个数字相加必须严格等于 1.0）
SPLIT_RATIOS = {'train': 0.85, 'val': 0.1, 'test': 0.05}


def preprocess_and_split():
    """
    核心调度流水线：遍历原始目录 -> 随机抓阄 -> 清洗压缩 -> 落盘保存
    """
    # 1. 扫描出原数据目录下所有的合法类别文件夹（例如 '0', 'a', 'H_caps'）
    categories = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]

    print("=" * 60)
    print(f"🚀 启动数据预处理引擎... 目标物理分辨率: {TARGET_SIZE}")
    print(f"📊 切分策略: Train {SPLIT_RATIOS['train']:.0%} | Val {SPLIT_RATIOS['val']:.0%} | Test {SPLIT_RATIOS['test']:.0%}")
    print("=" * 60)

    # 遍历每一个类别（字母或数字）
    for cat in categories:
        cat_in_path = os.path.join(RAW_DATA_DIR, cat)

        # 过滤筛选：只提取出标准的图像文件，防止系统中混入 .DS_Store 或 .txt 等杂质文件导致训练崩溃
        images = [img for img in os.listdir(cat_in_path) if img.endswith(('.png', '.jpg', '.jpeg'))]

        # tqdm 包裹 images 列表，在终端渲染出该类别的处理进度条
        for img_name in tqdm(images, desc=f"正在清洗类别 [{cat:>6s}]"):

            # ------------------------------------------------------------
            # 步骤 A：【增量断点续传检查】(Incremental Check)
            # ------------------------------------------------------------
            # 作用：如果处理几十万张图时电脑死机了，下次重启只需跳过已处理的，秒出进度。
            is_processed = False
            for split_name in SPLIT_RATIOS.keys():
                # 拼装假设它已经被处理过了的物理路径
                check_path = os.path.join(OUTPUT_DIR, split_name, cat, img_name)
                # 如果这个文件在硬盘上客观存在，说明它之前被处理过
                if os.path.exists(check_path):
                    is_processed = True
                    break

            # 如果已经被处理过，直接跳过后面的全部逻辑，处理下一张图
            if is_processed:
                continue

            # ------------------------------------------------------------
            # 步骤 B：【随机轮盘赌分配】(Random Routing)
            # ------------------------------------------------------------
            # random.random() 会生成一个 [0.0, 1.0) 之间的均匀随机小数
            rand_val = random.random()

            # 根据随机数落入的区间，决定这张图片的命运：
            # 区间 [0.0, 0.85) -> 划入训练集
            if rand_val < SPLIT_RATIOS['train']:
                target_split = 'train'
            # 区间 [0.85, 0.95) -> 划入验证集
            elif rand_val < SPLIT_RATIOS['train'] + SPLIT_RATIOS['val']:
                target_split = 'val'
            # 区间 [0.95, 1.0) -> 划入测试集
            else:
                target_split = 'test'

            # ------------------------------------------------------------
            # 步骤 C：【图像光学清洗与落盘】(Image Processing)
            # ------------------------------------------------------------
            # 动态生成最终要存放的具体文件夹路径 (例如: data/processed/train/A_caps)
            out_path = os.path.join(OUTPUT_DIR, target_split, cat)
            os.makedirs(out_path, exist_ok=True) # exist_ok=True 保证如果目录已存在不会报错

            # 最终文件的完整落盘路径
            out_file_path = os.path.join(out_path, img_name)

            # 使用 OpenCV 读取内存
            img = cv2.imread(os.path.join(cat_in_path, img_name))
            # 容错机制：如果图片损坏（体积为0或无法解码），直接抛弃
            if img is None:
                continue

            # 【核心洗图 1】：色彩剥离 (RGB to Grayscale)
            # 将彩色图转为单通道灰度图。字符识别只看重“形状轮廓”，颜色全是干扰噪声，
            # 转成灰度图能直接将模型计算量降低 3 倍！
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 【核心洗图 2】：空间降维 (Resize)
            # interpolation=cv2.INTER_AREA (区域插值法)：
            # 这是 OpenCV 中缩小图像时的【最优解】，能最大程度避免摩尔纹和锯齿现象，让字符边缘平滑。
            img_resized = cv2.resize(img_gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)

            # 物理写入硬盘
            cv2.imwrite(out_file_path, img_resized)

    print("\n" + "=" * 60)
    print(f"✅ 全量数据清洗与物理隔离完成！")
    print(f"📦 标准化数据沙箱已输出至：{OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    # ============================================================
    # 随机数种子注入 (Random Seed Injection)
    # ============================================================
    # 【工程化细节】：在严谨的科学实验中，我们希望所有的随机都是可以复现的。
    # 如果取消下面这行的注释，那么无论你在什么电脑上跑多少次，
    # 第一张图永远会被分到训练集，第二张永远在验证集... 保证了多人协作时数据集的一致性。
    # random.seed(42)

    preprocess_and_split()