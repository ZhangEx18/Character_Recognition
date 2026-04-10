"""
数据预处理与切分脚本 (放置于 tools 文件夹)
运行方式: 确保终端当前路径为【项目根目录】，运行: python -m tools.preprocess
"""
import os
import cv2
import random
from tqdm import tqdm

# ================= 配置区 =================
RAW_DATA_DIR = 'data/raw/archive/augmented_images/augmented_images1'
OUTPUT_DIR = 'data/processed'
TARGET_SIZE = (64, 64)

# 数据集划分比例 (训练集:验证集:测试集)
# 确保三个数字相加等于 1.0
SPLIT_RATIOS = {'train': 0.85, 'val': 0.1, 'test': 0.05}

def preprocess_and_split():
    categories = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]

    print(f"🚀 开始处理并切分数据集... 目标分辨率: {TARGET_SIZE}")

    for cat in categories:
        cat_in_path = os.path.join(RAW_DATA_DIR, cat)

        # 获取所有有效图片
        images = [img for img in os.listdir(cat_in_path) if img.endswith(('.png', '.jpg', '.jpeg'))]

        for img_name in tqdm(images, desc=f"处理类别 {cat}"):
            # 1. 【增量逻辑检查】：检查该图片是否已经在任意一个集合中存在
            is_processed = False
            for split_name in SPLIT_RATIOS.keys():
                check_path = os.path.join(OUTPUT_DIR, split_name, cat, img_name)
                if os.path.exists(check_path):
                    is_processed = True
                    break

            # 如果在任何一个集合里找到了这张图，说明处理过了，直接跳过
            if is_processed:
                continue

            # 2. 【随机分配逻辑】：生成 0~1 之间的随机数，决定该图片的去向
            rand_val = random.random()
            if rand_val < SPLIT_RATIOS['train']:
                target_split = 'train'
            elif rand_val < SPLIT_RATIOS['train'] + SPLIT_RATIOS['val']:
                target_split = 'val'
            else:
                target_split = 'test'

            # 3. 创建目标路径并处理图像
            out_path = os.path.join(OUTPUT_DIR, target_split, cat)
            os.makedirs(out_path, exist_ok=True)
            out_file_path = os.path.join(out_path, img_name)

            img = cv2.imread(os.path.join(cat_in_path, img_name))
            if img is None:
                continue

            # 转灰度并缩放
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)

            cv2.imwrite(out_file_path, img_resized)

    print(f"\n✅ 处理与划分完成！数据已输出至：{OUTPUT_DIR}")

if __name__ == "__main__":
    # 可选：设置随机种子，让每次随机划分的结果可复现
    # random.seed(42)
    preprocess_and_split()