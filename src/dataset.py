"""
数据加载与增强模块 (Data Loading & Augmentation Pipeline)

【系统架构定位】
本模块是整个 CNN 字符识别系统的“数据咽喉”。神经网络的上限由数据决定，本模块负责
将冰冷的硬盘文件转化为网络可以直接吞吐的张量 (Tensor) 血液。

【核心职责】
1. 类别解析：自动映射数字 (0-9)、小写字母 (a-z) 与大写字母 (A-Z) 共 62 个分类。
2. 性能榨取：采用“牺牲内存换取极致 I/O 速度”的策略，将图片全量缓存至 RAM。
3. 数据增强：在训练时引入随机仿射、旋转，模拟真实世界中潦草、倾斜的手写轨迹。
4. 环境沙箱：严格隔离 Train / Val / Test 数据集，杜绝数据泄露导致的“假高分”。
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Callable
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


# ============================================================
# 1. 类别映射字典的构建与全局常量
# ============================================================

def build_class_mapping() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    构建类别名称 (String) 与网络输出索引 (Integer) 的双向映射字典。

    【为什么需要这个？】
    神经网络的交叉熵损失函数 (CrossEntropyLoss) 不认识字符串 'A' 或 'b'，
    它只认识 0 到 61 之间的整数标签。我们需要一个标准化的翻译官。

    返回:
        name_to_idx: 例如 {'0': 0, 'a': 10, 'A': 36}
        idx_to_name: 例如 {0: '0', 10: 'a', 36: 'A'}
    """
    name_to_idx = {}
    idx_to_name = {}

    # 1. 注入数字 0-9 (索引范围 0-9)
    for num_val in range(10):
       name = str(num_val)
       name_to_idx[name] = num_val
       idx_to_name[num_val] = name

    # 2. 注入小写字母 a-z (索引范围 10-35)
    for offset, char in enumerate('abcdefghijklmnopqrstuvwxyz'):
       idx = 10 + offset
       name_to_idx[char] = idx
       idx_to_name[idx] = char

    # 3. 注入大写字母 A-Z (索引范围 36-61)
    for offset, char in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
       idx = 36 + offset
       name_to_idx[char] = idx
       idx_to_name[idx] = char

    return name_to_idx, idx_to_name


# 【全局单例】: 在模块导入时立即执行一次，生成供整个项目共享的常量字典
CLASS_NAME_TO_IDX, CLASS_IDX_TO_NAME = build_class_mapping()
NUM_CLASSES = len(CLASS_NAME_TO_IDX)  # 固定为 62 类


# ============================================================
# 2. 核心数据集类定义 (PyTorch Dataset 规范实现)
# ============================================================

class CharacterDataset(Dataset):
    """
    高度优化的自定义字符数据集。
    继承自 torch.utils.data.Dataset，重写了 __len__ 和 __getitem__ 魔术方法。
    """

    def __init__(
          self,
          data_dir: str,
          transform: Optional[Callable] = None,
          image_size: int = 64
    ):
       """
       初始化数据集上下文。

       参数:
           data_dir: 数据所在的物理根目录 (如 'data/processed/train')
           transform: 预处理与增强流水线函数
           image_size: 图像标准裁剪尺寸
       """
       self.data_dir = Path(data_dir)
       self.transform = transform
       self.image_size = image_size

       # samples 存储元组: (图片物理路径, 对应的整数标签)
       self.samples: List[Tuple[str, int]] = []

       # cached_images 存储真实的 PIL 图像对象，用于内存加速
       self.cached_images = []

       # 实例化时立即触发数据解析与内存装载
       self._load_samples()

    def _load_samples(self):
       """
       遍历物理目录树，解析类别并执行全量内存加载。
       【工程权衡】：对于几万张 64x64 的小图片，从硬盘频繁读取会造成极其严重的 I/O 瓶颈。
       我们选择在初始化时一次性将它们全部读入内存 (RAM)，让 GPU 训练时不需要等待硬盘。
       """
       if not self.data_dir.exists():
          raise FileNotFoundError(f"致命错误：数据目录不存在: {self.data_dir}")

       # 第一步：遍历目录结构，收集路径和标签
       for class_dir in self.data_dir.iterdir():
          if not class_dir.is_dir(): continue

          # 提取文件夹名称并翻译为整数标签
          class_idx = self._parse_class_name(class_dir.name)
          if class_idx is None: continue

          # 抓取所有支持的图片格式
          for ext in ["*.png", "*.jpg", "*.jpeg"]:
             for img_path in class_dir.glob(ext):
                self.samples.append((str(img_path), class_idx))

       print(f"成功扫描到 {len(self.samples)} 张有效图像。")

       # 第二步：将图片物理读入内存池
       print("🚀 正在将数据全量装载至内存池（以牺牲一定 RAM 换取极致训练速度）...")
       for img_path, _ in tqdm(self.samples, desc="装载进度"):
          # 使用 'L' 模式强制转换为 8-bit 单通道灰度图，大幅压缩内存占用
          img = Image.open(img_path).convert('L')
          self.cached_images.append(img)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
       """
       数据分发核心逻辑。 DataLoader 每次通过索引向这里要数据。

       参数:
           idx: 样本序号

       返回:
           (经过处理的图像张量 Tensor, 整数标签 Integer)
       """
       # 1. 直接从高速缓存中取图，而不是去读硬盘
       img = self.cached_images[idx]
       label = self.samples[idx][1]

       # 2. 动态应用数据增强和归一化（每次拿到的增强效果都不一样）
       if self.transform:
          img = self.transform(img)

       return img, label

    def __len__(self) -> int:
       """返回数据集总容量"""
       return len(self.samples)

    @staticmethod
    def _parse_class_name(dir_name: str) -> Optional[int]:
       """
       目录名安全解析器。
       处理类似 '0', 'a', 以及避免 Windows 不区分大小写而引入的 'A_caps' 等特殊后缀。
       """
       # 匹配数字 '0' - '9'
       if dir_name.isdigit():
          return int(dir_name)

       # 匹配单字母小写 'a' - 'z'
       if len(dir_name) == 1 and dir_name.islower():
          return CLASS_NAME_TO_IDX.get(dir_name)

       # 匹配带有 '_caps' 后缀的大写字母文件夹 (例如 'H_caps' 提取出 'H')
       if dir_name.endswith("_caps"):
          letter = dir_name[0].upper()
          if len(letter) == 1 and letter.isupper():
             return CLASS_NAME_TO_IDX.get(letter)

       return None

    def get_class_distribution(self) -> Dict[int, int]:
       """
       统计当前数据集中每个类别的样本数量。
       用于排查是否存在“数据倾斜 (Data Imbalance)”问题。
       """
       distribution = {}
       for _, label in self.samples:
          distribution[label] = distribution.get(label, 0) + 1
       return distribution


# ============================================================
# 3. 计算机视觉变换与数据增强流水线
# ============================================================

def get_train_transform(image_size: int = 64) -> transforms.Compose:
    """
    训练集专用的数据增强流水线。
    目的：通过人造的随机干扰，逼迫网络学习字符的核心骨架，而不是死记硬背像素位置。
    """
    return transforms.Compose([
       transforms.Resize((image_size, image_size)), # 统一尺寸
       # 1. 随机正负 15 度旋转，模拟书写时的倾斜
       transforms.RandomRotation(degrees=15),
       # 2. 随机仿射：上下左右平移最多 10%，大小随机缩放至 90%-110% 之间
       transforms.RandomAffine(
          degrees=0,
          translate=(0.1, 0.1),
          scale=(0.9, 1.1)
       ),
       # 3. 将 PIL Image [0, 255] 转换为 FloatTensor [0.0, 1.0]，并在最前面增加颜色通道维度
       transforms.ToTensor(),
       # 4. 数学归一化：将 [0.0, 1.0] 映射到 [-1.0, 1.0]。公式: (x - 0.5) / 0.5
       # 作用是让激活函数更快收敛，并减缓梯度消失
       transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def get_test_transform(image_size: int = 64) -> transforms.Compose:
    """
    验证集与测试集专用的流水线。
    【黄金法则】：测试时绝对不能有任何随机性（如旋转缩放），必须原汁原味地喂给模型评估。
    """
    return transforms.Compose([
       transforms.Resize((image_size, image_size)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.5], std=[0.5])
    ])


# ============================================================
# 4. 数据分发器 (DataLoader) 编排中心
# ============================================================

def create_dataloaders(
       data_dir: str,
       batch_size: int = 128,
       image_size: int = 64,
       num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    工厂函数：一键生成模型训练所需的三大数据流。

    参数:
        data_dir: 包含 train/val/test 的处理后根目录
        batch_size: 每批次吞吐的样本量 (影响显存/内存占用和梯度平滑度)
        image_size: 喂入网络的图片尺寸
        num_workers: 多线程读取数量 (由于我们使用了 RAM 全量缓存，设为 0 即可，多线程反而增加开销)
    """
    # 拼装三个子集的绝对路径
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # 初始化 Dataset 沙箱
    train_dataset = CharacterDataset(train_dir, get_train_transform(image_size), image_size)
    val_dataset = CharacterDataset(val_dir, get_test_transform(image_size), image_size)
    test_dataset = CharacterDataset(test_dir, get_test_transform(image_size), image_size)

    # 包装为可迭代的 DataLoader
    # 训练集必须打乱 (shuffle=True) 以防止网络记住排序规律
    train_loader = DataLoader(
       train_dataset, batch_size=batch_size, shuffle=True,
       num_workers=num_workers, pin_memory=False
    )
    # 验证集和测试集不需要打乱
    val_loader = DataLoader(
       val_dataset, batch_size=batch_size, shuffle=False,
       num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
       test_dataset, batch_size=batch_size, shuffle=False,
       num_workers=num_workers, pin_memory=False
    )

    print(f"\n✅ 数据集沙箱构建完成：")
    print(f"  训练集 (Train) : {len(train_dataset)} 张 (开启数据增强随机变异)")
    print(f"  验证集 (Val)   : {len(val_dataset)} 张 (纯净标准流，用于监控过拟合)")
    print(f"  测试集 (Test)  : {len(test_dataset)} 张 (纯净标准流，用于最终防作弊盲测)")

    return train_loader, val_loader, test_loader


# ============================================================
# 5. 数据健康度与增强效果可视化监测工具
# ============================================================

def visualize_samples(
       dataset: Dataset,
       num_samples: int = 16,
       save_path: Optional[str] = None
):
    """
    抽取数据集中的随机样本并绘制成网格图。
    主要用于在训练前人类用肉眼检查“数据增强是否过于离谱”或“标签是否错位”。
    """
    import matplotlib.pyplot as plt
    import random
    import matplotlib
    matplotlib.use('Agg') # 强制使用无 GUI 的后端绘制，防止服务器环境下报错

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
       if i >= num_samples: break

       idx = random.randint(0, len(dataset) - 1)
       img, label = dataset[idx]

       # 反归一化：由于前面 Normalize 把数值变成了 [-1, 1]
       # matplotlib 画图时需要将其恢复为 [0, 1] 的颜色范围，否则画面全黑或报错
       img = img * 0.5 + 0.5

       # squeeze() 去除 channel 维度：(1, 64, 64) -> (64, 64) 以满足 pyplot 的灰度图要求
       ax.imshow(img.squeeze().numpy(), cmap='gray')
       ax.set_title(f'Label: {CLASS_IDX_TO_NAME[label]}')
       ax.axis('off')

    plt.tight_layout()

    if save_path:
       plt.savefig(save_path, dpi=150)
       print(f"样本特征可视化拼图已落盘保存至: {save_path}")
    else:
       plt.show()


# ============================================================
# 测试桩 (Mock Test)
# 当直接运行 `python src/dataset.py` 时执行此代码块进行模块自检
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Dataset & DataLoader 模块自检程序")
    print("=" * 60)

    print("\n类别字典双向映射装载成功 (共 62 类).")

    # 模拟指向处理好的训练集目录
    test_target_dir = "../data/processed/train"

    if os.path.exists(test_target_dir):
       print(f"\n挂载并测试目标目录: {test_target_dir}")

       # 模拟构建 Dataset
       test_ds = CharacterDataset(
           data_dir=test_target_dir,
           transform=get_train_transform(64)
       )

       print(f"\n整体数据集容量: {len(test_ds)}")

       # 检查类别分布情况
       dist_info = test_ds.get_class_distribution()
       print(f"\n前10个类别的频率直方分布:")
       for class_id in range(10):
          count_val = dist_info.get(class_id, 0)
          print(f"  {CLASS_IDX_TO_NAME[class_id]:>2s} : {count_val:>5d} 张")

       # 抽查样本图像并保存
       visualize_samples(test_ds, save_path="dataset_augmented_samples.png")
    else:
       print(f"\n[阻断]: 目标训练集目录缺失: {test_target_dir}")
       print("提示：请先在项目根目录运行 python -m tools.preprocess 准备物理数据")

    print("\n" + "=" * 60)
    print("模块自检运行完毕！")
    print("=" * 60)