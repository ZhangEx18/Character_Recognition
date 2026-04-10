"""
数据预处理脚本 (放置于 tools 文件夹)
运行方式: python -m tools.preprocess
"""
import os
import cv2
from tqdm import tqdm

# 【核心修改】：抛弃长串的绝对路径，直接相对于根目录！
RAW_DATA_DIR = 'data/raw/archive/augmented_images/augmented_images1'
OUTPUT_DIR = 'data/processed'
TARGET_SIZE = (64, 64)

def preprocess():
    if not os.path.exists(OUTPUT_DIR):
       os.makedirs(OUTPUT_DIR)

    categories = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]

    print(f"🚀 开始增量预处理：新图片将缩放至 {TARGET_SIZE}")
    # ... 后续代码完全不变 ...
    for cat in categories:
       cat_in_path = os.path.join(RAW_DATA_DIR, cat)
       cat_out_path = os.path.join(OUTPUT_DIR, cat)
       os.makedirs(cat_out_path, exist_ok=True)

       for img_name in tqdm(os.listdir(cat_in_path), desc=f"处理类别 {cat}"):
          if not img_name.endswith(('.png', '.jpg', '.jpeg')):
             continue

          # 核心修改：定义输出文件路径并检查是否已存在
          out_file_path = os.path.join(cat_out_path, img_name)
          if os.path.exists(out_file_path):
             continue  # 如果目标文件已存在，则跳过，只针对新素材转换

          img = cv2.imread(os.path.join(cat_in_path, img_name))
          if img is None: continue

          # 转为灰度图 + 缩放
          img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          img_resized = cv2.resize(img_gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)

          cv2.imwrite(out_file_path, img_resized)

    print(f"\n✅ 增量预处理完成！数据已就绪：{OUTPUT_DIR}")


if __name__ == "__main__":
    preprocess()