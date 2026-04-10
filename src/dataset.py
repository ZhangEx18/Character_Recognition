"""
数据加载与增强模块 (Data Loading & Augmentation Pipeline)

负责：
1. 解析并加载字符识别数据集（数字 0-9 + 小写 a-z + 大写 A-Z 共 62 类）
2. 极限性能优化（基于 RAM 的全量内存缓存机制）
3. 图像预处理与张量标准化处理
4. 数据增强（随机旋转、平移、缩放等防止过拟合）
5. 科学划分数据集（Train / Validation / Test 严格隔离）
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Callable
from PIL import Image

import torch
# 【修复】：移除了未使用的 random_split 和 Subset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


# ============================================================
# 类别映射字典的构建
# ============================================================

def build_class_mapping() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    构建类别名称(String)与网络输出索引(Integer)的双向映射
    """
    name_to_idx = {}
    idx_to_name = {}

    # 1. 注入数字 0-9
    for num_val in range(10):
       name = str(num_val)
       name_to_idx[name] = num_val
       idx_to_name[num_val] = name

    # 2. 注入小写字母 a-z (拼写警告可安全忽略)
    for offset, char in enumerate('abcdefghijklmnopqrstuvwxyz'):
       idx = 10 + offset
       name_to_idx[char] = idx
       idx_to_name[idx] = char

    # 3. 注入大写字母 A-Z (拼写警告可安全忽略)
    for offset, char in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
       idx = 36 + offset
       name_to_idx[char] = idx
       idx_to_name[idx] = char

    return name_to_idx, idx_to_name


# 全局类别映射常量，供整个项目调用
CLASS_NAME_TO_IDX, CLASS_IDX_TO_NAME = build_class_mapping()
NUM_CLASSES = len(CLASS_NAME_TO_IDX)  # 62


# ============================================================
# 核心数据集类定义
# ============================================================

class CharacterDataset(Dataset):
    """
    高度优化的自定义字符数据集
    """

    def __init__(
          self,
          data_dir: str,
          transform: Optional[Callable] = None,
          image_size: int = 64
    ):
       self.data_dir = Path(data_dir)
       self.transform = transform
       self.image_size = image_size

       self.samples: List[Tuple[str, int]] = []
       self.cached_images = []

       self._load_samples()

    def _load_samples(self):
       """遍历目录树，解析类别并执行全量内存加载"""
       if not self.data_dir.exists():
          raise FileNotFoundError(f"致命错误：数据目录不存在: {self.data_dir}")

       for class_dir in self.data_dir.iterdir():
          if not class_dir.is_dir(): continue

          class_idx = self._parse_class_name(class_dir.name)
          if class_idx is None: continue

          for ext in ["*.png", "*.jpg", "*.jpeg"]:
             for img_path in class_dir.glob(ext):
                self.samples.append((str(img_path), class_idx))

       print(f"成功扫描到 {len(self.samples)} 张有效图像。")

       print("🚀 正在将数据全量装载至内存池（以牺牲一定 RAM 换取极致训练速度）...")
       for img_path, _ in tqdm(self.samples, desc="装载进度"):
          img = Image.open(img_path).convert('L')
          self.cached_images.append(img)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
       img = self.cached_images[idx]
       label = self.samples[idx][1]

       if self.transform:
          img = self.transform(img)

       return img, label

    def __len__(self) -> int:
       return len(self.samples)

    # 【修复】：添加了 staticmethod 装饰器，符合规范
    @staticmethod
    def _parse_class_name(dir_name: str) -> Optional[int]:
       """目录名安全解析器"""
       if dir_name.isdigit():
          return int(dir_name)

       if len(dir_name) == 1 and dir_name.islower():
          return CLASS_NAME_TO_IDX.get(dir_name)

       if dir_name.endswith("_caps"):
          letter = dir_name[0].upper()
          if len(letter) == 1 and letter.isupper():
             return CLASS_NAME_TO_IDX.get(letter)

       return None

    def get_class_distribution(self) -> Dict[int, int]:
       distribution = {}
       for _, label in self.samples:
          distribution[label] = distribution.get(label, 0) + 1
       return distribution


# ============================================================
# 计算机视觉变换与数据增强流水线
# ============================================================

def get_train_transform(image_size: int = 64) -> transforms.Compose:
    return transforms.Compose([
       transforms.Resize((image_size, image_size)),
       transforms.RandomRotation(degrees=15),
       transforms.RandomAffine(
          degrees=0,
          translate=(0.1, 0.1),
          scale=(0.9, 1.1)
       ),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def get_test_transform(image_size: int = 64) -> transforms.Compose:
    return transforms.Compose([
       transforms.Resize((image_size, image_size)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.5], std=[0.5])
    ])


# ============================================================
# 数据分发器 (DataLoader) 编排
# ============================================================

def create_dataloaders(
       data_dir: str,
       batch_size: int = 128,
       image_size: int = 64,
       num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    train_dataset = CharacterDataset(
       data_dir=train_dir,
       transform=get_train_transform(image_size),
       image_size=image_size
    )

    val_dataset = CharacterDataset(
       data_dir=val_dir,
       transform=get_test_transform(image_size),
       image_size=image_size
    )

    test_dataset = CharacterDataset(
       data_dir=test_dir,
       transform=get_test_transform(image_size),
       image_size=image_size
    )

    train_loader = DataLoader(
       train_dataset,
       batch_size=batch_size,
       shuffle=True,
       num_workers=num_workers,
       pin_memory=False
    )

    val_loader = DataLoader(
       val_dataset,
       batch_size=batch_size,
       shuffle=False,
       num_workers=num_workers,
       pin_memory=False
    )

    test_loader = DataLoader(
       test_dataset,
       batch_size=batch_size,
       shuffle=False,
       num_workers=num_workers,
       pin_memory=False
    )

    print(f"\n✅ 数据集沙箱构建完成：")
    print(f"  训练集 (Train) : {len(train_dataset)} 张 (附带数据增强)")
    print(f"  验证集 (Val)   : {len(val_dataset)} 张 (纯净标准流)")
    print(f"  测试集 (Test)  : {len(test_dataset)} 张 (纯净标准流)")

    return train_loader, val_loader, test_loader

# ============================================================
# 数据健康度监测工具
# ============================================================

def visualize_samples(
       dataset: Dataset,
       num_samples: int = 16,
       save_path: Optional[str] = None
):
    import matplotlib.pyplot as plt
    import random
    import matplotlib
    matplotlib.use('Agg')

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    # 【修复】：使用了 num_samples 变量，解除了未使用警告
    for i, ax in enumerate(axes.flat):
       if i >= num_samples: break

       idx = random.randint(0, len(dataset) - 1)
       img, label = dataset[idx]

       img = img * 0.5 + 0.5

       ax.imshow(img.squeeze().numpy(), cmap='gray')
       ax.set_title(f'标签: {CLASS_IDX_TO_NAME[label]}')
       ax.axis('off')

    plt.tight_layout()

    if save_path:
       plt.savefig(save_path, dpi=150)
       print(f"样本特征可视化拼图已落盘保存至: {save_path}")
    else:
       plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("Dataset & DataLoader 模块自检程序")
    print("=" * 60)

    print("\n类别字典双向映射装载成功 (共 62 类).")

    # 【修复】：重写了测试代码逻辑。现在它去加载您的 train 文件夹，并删除了 TransformSubset
    test_target_dir = "../data/processed/train"

    if os.path.exists(test_target_dir):
       print(f"\n挂载并测试目标目录: {test_target_dir}")

       # 直接在初始化时传入数据增强，无需包装
       test_ds = CharacterDataset(
           data_dir=test_target_dir,
           transform=get_train_transform(64)
       )

       print(f"\n整体数据集容量: {len(test_ds)}")

       dist_info = test_ds.get_class_distribution()
       print(f"\n前10个类别的频率直方分布:")
       for class_id in range(10):  # 避免变量名冲突
          count_val = dist_info.get(class_id, 0)
          print(f"  {CLASS_IDX_TO_NAME[class_id]:>2s} : {count_val:>5d} 张")

       visualize_samples(test_ds, save_path="dataset_augmented_samples.png")
    else:
       print(f"\n[阻断]: 目标训练集目录缺失: {test_target_dir}")
       print("提示：请先在项目根目录运行 python -m tools.preprocess")

    print("\n" + "=" * 60)
    print("模块自检运行完毕！")
    print("=" * 60)