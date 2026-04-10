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
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from tqdm import tqdm


# ============================================================
# 类别映射字典的构建
# ============================================================

def build_class_mapping() -> Tuple[Dict[str, int], Dict[int, str]]:
	"""
    构建类别名称(String)与网络输出索引(Integer)的双向映射

    分类总计 62 个类别，排列顺序极其重要（与推断时的字典强绑定）：
    - [0, 9]   : 数字 '0' 到 '9'
    - [10, 35] : 小写字母 'a' 到 'z'
    - [36, 61] : 大写字母 'A' 到 'Z'

    返回:
        name_to_idx: 类别名称 → 索引 (例如：{'a': 10})
        idx_to_name: 索引 → 类别名称 (例如：{10: 'a'})
    """
	name_to_idx = {}
	idx_to_name = {}

	# 1. 注入数字 0-9
	for i in range(10):
		name = str(i)
		name_to_idx[name] = i
		idx_to_name[i] = name

	# 2. 注入小写字母 a-z
	for i, char in enumerate('abcdefghijklmnopqrstuvwxyz'):
		idx = 10 + i
		name_to_idx[char] = idx
		idx_to_name[idx] = char

	# 3. 注入大写字母 A-Z
	for i, char in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
		idx = 36 + i
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

    【性能优化核心】：
    传统的 Dataset 是在 __getitem__ 中调用 Image.open() 读取硬盘。
    当数据量大且 GPU 运算极快时，硬盘 I/O 会成为严重瓶颈（GPU等CPU，CPU等硬盘）。
    此类在初始化时将所有图片一次性读入 RAM (cached_images)，后续训练直接从内存切片，
    极大提升 Epoch 迭代速度。
    """

	def __init__(
			self,
			data_dir: str,
			transform: Optional[Callable] = None,
			image_size: int = 64  # 匹配最新的 64x64 网络配置
	):
		self.data_dir = Path(data_dir)
		self.transform = transform
		self.image_size = image_size

		# 1. 初始化容器
		self.samples: List[Tuple[str, int]] = []  # 存储 (文件路径, 类别索引) 记录
		self.cached_images = []  # 存储真实 PIL 图像对象的内存池

		# 2. 扫描路径并立即读入内存池
		self._load_samples()

	def _load_samples(self):
		"""遍历目录树，解析类别并执行全量内存加载"""
		if not self.data_dir.exists():
			raise FileNotFoundError(f"致命错误：数据目录不存在: {self.data_dir}")

		# 扫描并解析各个类别文件夹
		for class_dir in self.data_dir.iterdir():
			if not class_dir.is_dir(): continue

			# 利用独立函数解析文件夹名称，防止因命名不规范导致的错位
			class_idx = self._parse_class_name(class_dir.name)
			if class_idx is None: continue

			for ext in ["*.png", "*.jpg", "*.jpeg"]:
				for img_path in class_dir.glob(ext):
					self.samples.append((str(img_path), class_idx))

		print(f"成功扫描到 {len(self.samples)} 张有效图像。")

		# 核心逻辑：全量加载至内存
		print("🚀 正在将数据全量装载至内存池（以牺牲一定 RAM 换取极致训练速度）...")
		for img_path, _ in tqdm(self.samples, desc="装载进度"):
			# 预先读取并转为灰度图 ('L' 模式)，防止 RGB 图像占用过多内存，随后存入列表
			img = Image.open(img_path).convert('L')
			self.cached_images.append(img)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
		"""
        数据获取器：训练大循环中 DataLoader 每次调用的核心函数
        """
		# 【关键加速点】：不再执行高耗时的 Image.open(path)，直接 O(1) 从内存池提取
		img = self.cached_images[idx]
		label = self.samples[idx][1]

		# 动态应用变换（注意：数据增强必须在这里动态应用，而不是存入内存时。
		# 这样才能保证每个 Epoch 抽到的同一张图都有不同的随机旋转/缩放效果）
		if self.transform:
			img = self.transform(img)

		return img, label

	def __len__(self) -> int:
		return len(self.samples)

	def _parse_class_name(self, dir_name: str) -> Optional[int]:
		"""
        目录名安全解析器
        规则：
        - "0" ~ "9" → 映射至数字索引 0-9
        - "a" ~ "z" → 映射至小写字母索引 10-35
        - "A_caps" ~ "Z_caps" → 映射至大写字母索引 36-61 (防止 Windows 大小写不敏感问题)
        """
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
		"""统计数据集中各类别分布情况，用于排查数据不平衡问题"""
		distribution = {}
		for _, label in self.samples:
			distribution[label] = distribution.get(label, 0) + 1
		return distribution


class TransformSubset(Dataset):
	"""
    轻量级的数据集包装器（防踩坑利器）

    【解决痛点】：PyTorch 的 random_split 返回的 Subset 会直接共享原始 Dataset。
    如果直接修改 Subset 的 transform，会污染整个原始数据集。
    此类负责在提取数据时动态应用指定的 transform，实现内存中只有一份数据，
    但训练集、验证集可以有截然不同的预处理流。
    """

	def __init__(self, subset: Subset, transform: Callable):
		self.subset = subset
		self.transform = transform

	def __getitem__(self, idx):
		# 从底层的 Subset 的原始数据集中拉取未经 transform 的原始内存图像
		img = self.subset.dataset.cached_images[self.subset.indices[idx]]
		label = self.subset.dataset.samples[self.subset.indices[idx]][1]

		# 动态应用独立的预处理流
		if self.transform:
			img = self.transform(img)
		return img, label

	def __len__(self):
		return len(self.subset)


# ============================================================
# 计算机视觉变换与数据增强流水线
# ============================================================

def get_train_transform(image_size: int = 64) -> transforms.Compose:
	"""
    训练集专用增强管道

    作用：制造"合理的噪声"，强迫网络学习字符的核心骨架，而非死记硬背像素位置。
    """
	return transforms.Compose([
		transforms.Resize((image_size, image_size)),

		# --- 增强层 ---
		# 随机轻微旋转 (-15° ~ +15°)，模拟手写或文档扫描歪斜
		transforms.RandomRotation(degrees=15),

		# 随机仿射变换
		# translate: 图像可能上下左右平移最多 10%
		# scale: 图像可能缩放至原本的 90% 到 110% 之间
		transforms.RandomAffine(
			degrees=0,
			translate=(0.1, 0.1),
			scale=(0.9, 1.1)
		),

		# --- 格式化层 ---
		# 张量化：将像素范围 [0, 255] 除以 255 转化为 [0.0, 1.0]
		transforms.ToTensor(),
		# 归一化：公式为 (x - mean) / std，使像素范围映射至 [-1.0, 1.0]，加速梯度下降收敛
		transforms.Normalize(mean=[0.5], std=[0.5])
	])


def get_test_transform(image_size: int = 64) -> transforms.Compose:
	"""
    验证/测试集专用预处理管道

    【核心原则】：考试期间绝不能制造干扰！因此去除了所有数据增强逻辑，仅保留标准化。
    """
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
		batch_size: int = 128,  # 设置为合理的批次大小，防止显存 OOM
		val_split: float = 0.1,
		test_split: float = 0.1,
		image_size: int = 64,
		num_workers: int = 0  # 核心配置项
) -> Tuple[DataLoader, DataLoader, DataLoader]:
	"""
    完整的工业级数据集创建流水线
    """
	# 1. 挂载基底数据集（此时不传入任何 transform，保留最原始的 PIL Image）
	full_dataset = CharacterDataset(
		data_dir=data_dir,
		transform=None,
		image_size=image_size
	)

	# 2. 计算各部分体积 (默认 80% 训练, 10% 验证, 10% 测试)
	total_size = len(full_dataset)
	test_size = int(total_size * test_split)
	val_size = int(total_size * val_split)
	train_size = total_size - test_size - val_size

	# 3. 随机切片划分
	# manual_seed 极度重要：确保每次运行代码划分出的训练集和测试集完全一致，保证对照实验的严谨性
	train_subset, val_subset, test_subset = random_split(
		full_dataset,
		[train_size, val_size, test_size],
		generator=torch.Generator().manual_seed(42)
	)

	# 4. 利用包装器分别挂载不同的预处理流（解决数据增强泄露问题）
	train_dataset = TransformSubset(train_subset, get_train_transform(image_size))
	val_dataset = TransformSubset(val_subset, get_test_transform(image_size))
	test_dataset = TransformSubset(test_subset, get_test_transform(image_size))

	# 5. 封装为批量化加载器 DataLoader
	# 【关于 num_workers = 0 的工程考量】：
	# 因为本程序使用了内存缓存 (cached_images)，全部数据已存在于主进程的 RAM 中。
	# 在 PC 端（特别是 Windows 缺少完善的 fork 机制下），如果启用多进程 (num_workers > 0)，
	# PyTorch 可能会将整个内存缓存复制多份分发给子进程，导致内存爆炸式泄漏并死机。
	# 既然在内存中，单线程提取的速度已经绝对够快了，所以强制设为 0 最稳妥。

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,  # 仅训练集需要打乱，打破样本间的序列相关性
		num_workers=num_workers,
		pin_memory=False  # 如果未使用 GPU 则保持 False，否则设为 True 可加速转移
	)

	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,  # 验证和测试集无需打乱，固定顺序利于排查
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
	print(f"  训练集 (Train) : {train_size} 张 (附带数据增强)")
	print(f"  验证集 (Val)   : {val_size} 张 (纯净标准流)")
	print(f"  测试集 (Test)  : {test_size} 张 (纯净标准流)")

	return train_loader, val_loader, test_loader


# ============================================================
# 数据健康度监测工具
# ============================================================

def visualize_samples(
		dataset: Dataset,
		num_samples: int = 16,
		save_path: Optional[str] = None
):
	"""
    抽取样本并反向渲染

    作用：用于训练前人工确认数据增强的强度是否合适（例如旋转有没有导致字符超出画幅）
    """
	import matplotlib.pyplot as plt
	import random
	import matplotlib
	matplotlib.use('Agg')

	fig, axes = plt.subplots(4, 4, figsize=(10, 10))

	for ax in axes.flat:
		idx = random.randint(0, len(dataset) - 1)
		img, label = dataset[idx]

		# 核心：反归一化还原
		# 预处理时做了: output = (input - 0.5) / 0.5
		# 还原时需做:   input = output * 0.5 + 0.5
		img = img * 0.5 + 0.5

		# .squeeze() 去除单通道带来的多余维度：(1, 64, 64) -> (64, 64)
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
	# 本模块直接运行时的单元测试脚本
	print("=" * 60)
	print("Dataset & DataLoader 模块自检程序")
	print("=" * 60)

	# ... 简略测试输出映射关系
	print("\n类别字典双向映射装载成功 (共 62 类).")

	# 指定一个相对路径测试数据集加载
	data_dir = "../data/processed"

	if os.path.exists(data_dir):
		print(f"\n挂载并测试根目录: {data_dir}")
		dataset = CharacterDataset(data_dir)

		print(f"\n整体数据集容量: {len(dataset)}")

		dist = dataset.get_class_distribution()
		print(f"\n前10个类别的频率直方分布:")
		for i in range(10):
			count = dist.get(i, 0)
			print(f"  {CLASS_IDX_TO_NAME[i]:>2s} : {count:>5d} 张")

		# 包装一层增强流测试可视化功能
		train_wrapper = TransformSubset(
			Subset(dataset, range(len(dataset))),
			get_train_transform(64)
		)
		visualize_samples(train_wrapper, save_path="dataset_augmented_samples.png")
	else:
		print(f"\n[阻断]: 目标目录缺失: {data_dir}")
		print("请运行 preprocess 脚本或修正路径。")

	print("\n" + "=" * 60)
	print("模块自检运行完毕！")
	print("=" * 60)