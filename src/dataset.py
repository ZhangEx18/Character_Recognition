"""
数据加载与增强模块 (Data Loading & Augmentation Pipeline)

【系统架构定位】
本模块构筑了卷积神经网络的数据接入层 (Data Ingestion Layer)。
其核心职责是将持久化存储的离散图像实体，转换为适配张量计算引擎的高维连续内存数据，
并确立模型优化的数据分布基准。

【全景处理流程】
[起点] 物理存储层 (Disk): data/processed/{train,val,test}
  │
  ├─ 1. 索引构建与标签映射: 遍历目录拓扑，完成 62 类字符到离散整型标量 (0-61) 的映射。
  ├─ 2. 内存驻留缓存 (In-Memory Caching): 全量读取 PIL 图像至 RAM，消除训练期的磁盘 I/O 瓶颈。
  │
[动态采样流] DataLoader 迭代获取 Batch
  │
  ├─ [训练期 (Train)] 随机增强流水线 (Stochastic Augmentation)
  │   ├─ 3a. 空间几何变换: 应用随机旋转 (Rotation) 与仿射变换 (Affine)，提升对形态畸变的平移/旋转不变性。
  │   └─ 4a. 张量化与标准化: 投射至 [-1.0, 1.0] 数据分布，缓解网络层间的协变量偏移。
  │
  ├─ [评估期 (Val/Test)] 确定性推断 (Deterministic Evaluation)
  │   └─ 3b. 纯净数据流: 屏蔽所有随机几何扰动，仅执行张量转换与数学标准化操作。
  │
[终点] 计算图输入流 (Tensor): (Batch_Size, 1, 64, 64) 规范化张量矩阵及对应的 Target 向量

========================================================================
【核心架构职责】
1. 标签离散化编码：构建字符类别 (0-9, a-z, A-Z) 至整型索引空间的双向映射。
2. I/O 吞吐优化：利用高速内存缓存 (RAM Caching) 突破传统磁盘寻道与加载延迟。
3. 数据增强正则化：在训练流中注入受控的几何扰动，抑制过拟合，提升模型泛化能力。
4. 评估沙箱隔离：构建严格确定的评估数据流，防止信息泄露影响模型泛化能力的客观评测。
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
# 1. 类别映射空间的构建与全局常量定义
# ============================================================

def build_class_mapping() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    构建字符类别 (String) 与目标网络输出标量 (Integer) 的双向哈希映射。

    【架构意义】：
    标准交叉熵损失函数 (CrossEntropyLoss) 需接收一维的非负整型张量作为计算目标。
    此函数提供将物理语义标签转换为优化器可识别离散状态的标准化方法。

    返回:
        name_to_idx: 字符到索引的映射，例如 {'0': 0, 'a': 10, 'A': 36}
        idx_to_name: 索引到字符的映射，例如 {0: '0', 10: 'a', 36: 'A'}
    """
    name_to_idx = {}
    idx_to_name = {}

    # 1. 注入数字类别 0-9 (标量区间 0-9)
    for num_val in range(10):
        name = str(num_val)
        name_to_idx[name] = num_val
        idx_to_name[num_val] = name

    # 2. 注入小写字母类别 a-z (标量区间 10-35)
    for offset, char in enumerate('abcdefghijklmnopqrstuvwxyz'):
        idx = 10 + offset
        name_to_idx[char] = idx
        idx_to_name[idx] = char

    # 3. 注入大写字母类别 A-Z (标量区间 36-61)
    for offset, char in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        idx = 36 + offset
        name_to_idx[char] = idx
        idx_to_name[idx] = char

    return name_to_idx, idx_to_name


# 【全局单例机制】: 模块初始化时执行分配，生成供整个计算进程共享的常量字典
CLASS_NAME_TO_IDX, CLASS_IDX_TO_NAME = build_class_mapping()
NUM_CLASSES = len(CLASS_NAME_TO_IDX)  # 约束类别维度为 62


# ============================================================
# 2. 核心数据集类定义 (基于 PyTorch Dataset 规范)
# ============================================================

class CharacterDataset(Dataset):
    """
    基于全量内存驻留策略优化的高效字符数据集。
    继承自 torch.utils.data.Dataset，封装了底层数据寻址与加载逻辑。
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
            data_dir: 结构化数据所在的物理基目录
            transform: 数据预处理与增强的级联函数 (Pipeline)
            image_size: 张量统一的空间分辨率标准
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_size = image_size

        # samples 容器: 存储 (物理路径, 整型标签) 元组结构
        self.samples: List[Tuple[str, int]] = []

        # cached_images 容器: 存储反序列化后的 PIL 图像对象
        self.cached_images = []

        # 触发底层数据解析与内存预载
        self._load_samples()

    def _load_samples(self):
        """
        遍历目录拓扑，解析类别空间并执行全量内存加载。

        【工程权衡】：针对 64x64 规模的小分辨率图像，频繁的磁盘 I/O 将成为 GPU 调度的严重瓶颈。
        本架构采用预加载策略 (Pre-loading)，将高频访问数据驻留于 RAM，从而实现计算资源的满载运转。
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"[Error]: 目标寻址失败，物理目录不存在: {self.data_dir}")

        # 阶段一：扫描目录层次结构，构建路径与标签映射清单
        for class_dir in self.data_dir.iterdir():
            if not class_dir.is_dir(): continue

            class_idx = self._parse_class_name(class_dir.name)
            if class_idx is None: continue

            # 遍历并收集受支持的图像容器格式
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                for img_path in class_dir.glob(ext):
                    self.samples.append((str(img_path), class_idx))

        print(f"目录解析完成，检测到 {len(self.samples)} 条有效数据记录。")

        # 阶段二：执行 I/O 解码并挂载至内存池
        print("🚀 执行全量内存驻留策略 (In-Memory Loading)...")
        for img_path, _ in tqdm(self.samples, desc="预载进度"):
            # 应用 'L' 模式约束像素深度为 8-bit 单通道灰度，优化内存空间占用
            img = Image.open(img_path).convert('L')
            self.cached_images.append(img)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        定义数据分发核心逻辑。
        响应 DataLoader 的迭代索引请求，输出单实例张量数据。
        """
        # 1. 规避磁盘 I/O，直接从驻留缓存中提取图像实例
        img = self.cached_images[idx]
        label = self.samples[idx][1]

        # 2. 动态应用数据增强与标准化级联 (动态计算图输入的前置处理)
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self) -> int:
        """返回当前数据集实例包含的样本总规模"""
        return len(self.samples)

    @staticmethod
    def _parse_class_name(dir_name: str) -> Optional[int]:
        """
        健壮的目录名称解析器。
        兼容数字、标准小写字母，以及为规避操作系统大小写不敏感特性而附加后缀的大写字母标识。
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
        """
        统计类别频率直方分布。
        提供先验数据，用于检测数据集是否存在严峻的长尾分布或类别失衡 (Data Imbalance) 现象。
        """
        distribution = {}
        for _, label in self.samples:
            distribution[label] = distribution.get(label, 0) + 1
        return distribution


# ============================================================
# 3. 几何变换与数据增强逻辑 (Data Augmentation)
# ============================================================

def get_train_transform(image_size: int = 64) -> transforms.Compose:
    """
    配置训练流的数据增强级联器。

    【增强策略】：
    通过引入受控的空间几何扰动 (旋转、平移、缩放、随机擦除)，强制网络学习目标的全局拓扑结构特征，
    降低模型对局部像素分布的过度拟合倾向，显著提升泛化鲁棒性。
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        # 引入局域旋转变换，约束范围 [-15, 15] 度
        transforms.RandomRotation(degrees=15),
        # 引入仿射变换：平移约束在空间维度的 ±10% 内，尺度缩放约束在 90%~110% 区间
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        # 【新增】随机擦除 (Random Erasing)：
        # 在图像上随机涂黑一小块区域，逼迫网络不能只靠局部特征（如字母的上半圈）来识别，
        # 必须学会通过残缺的部分也能认出完整字符。这极大提升对污损、残缺字符的识别能力。
        transforms.RandomErasing(
            p=0.3,           # 30% 概率执行擦除
            scale=(0.02, 0.15),  # 擦除面积占图像的 2%~15%
            ratio=(0.3, 3.3),    # 擦除区域的宽高比范围
            value=0          # 擦除后填黑 (0 = 黑色)
        ),
        # 将 PIL Image 转换为 FloatTensor 并归一化至 [0.0, 1.0]，同时升维增加 Channel 维度
        transforms.ToTensor(),
        # 标准化映射至 [-1.0, 1.0] 区间，加速梯度下降效率
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def get_test_transform(image_size: int = 64) -> transforms.Compose:
    """
    配置评估流 (验证集/测试集) 的数据流水线。

    【执行约束】：
    评估阶段必须隔离所有随机干扰因素，确保模型性能验证过程具备严格的客观性与可复现性。
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


# ============================================================
# 4. 数据迭代器编排系统 (DataLoader Orchestration)
# ============================================================

def create_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    image_size: int = 64,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    数据集聚合与迭代器实例化接口。

    参数:
        data_dir: 包含标准划分层次 (train/val/test) 的根目录路径
        batch_size: 批次容量参数 (决定单次参数更新的样本基数与显存占用率)
        image_size: 统一网络输入的空间分辨率
        num_workers: 异步数据预取线程数 (当前采用全量内存池设计，配置为 0 以规避进程间通信开销)
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # 初始化 Dataset 实体并挂载相应的数据流管线
    train_dataset = CharacterDataset(train_dir, get_train_transform(image_size), image_size)
    val_dataset = CharacterDataset(val_dir, get_test_transform(image_size), image_size)
    test_dataset = CharacterDataset(test_dir, get_test_transform(image_size), image_size)

    # 封装迭代器：训练流必须开启随机重排 (shuffle=True)，以破坏批次间的顺序相关性
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    # 评估流配置为顺序抽取 (shuffle=False)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )

    print(f"\n✅ 数据集迭代器编排完成：")
    print(f"  Train Loader : {len(train_dataset)} 样本 (已注入随机几何增强策略)")
    print(f"  Val Loader   : {len(val_dataset)} 样本 (确定性标准推断流)")
    print(f"  Test Loader  : {len(test_dataset)} 样本 (独立隔离盲测数据流)")

    return train_loader, val_loader, test_loader


# ============================================================
# 5. 数据流定性监测与可视化诊断组件
# ============================================================

def visualize_samples(
    dataset: Dataset,
    num_samples: int = 16,
    save_path: Optional[str] = None
):
    """
    抽样数据集内容并渲染特征网格阵列。
    主要用于在实验前期，定性验证增强策略的空间畸变幅度是否合理，以及标签对齐状态是否健康。
    """
    import matplotlib.pyplot as plt
    import random
    import matplotlib
    matplotlib.use('Agg')  # 绑定非交互式渲染后端，规避 X-Server 依赖引发的运行时异常

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        if i >= num_samples: break

        idx = random.randint(0, len(dataset) - 1)
        img, label = dataset[idx]

        # 逆向标准化：将张量数据由 [-1.0, 1.0] 线性重映射至标准视觉空间 [0.0, 1.0]
        img = img * 0.5 + 0.5

        # 执行张量降维 (Squeeze): 移除 Channel 维度适配单通道渲染引擎
        ax.imshow(img.squeeze().numpy(), cmap='gray')
        ax.set_title(f'Label: {CLASS_IDX_TO_NAME[label]}')
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"定性抽样特征矩阵已输出至指定路径: {save_path}")
    else:
        plt.show()


# ============================================================
# 模块级集成自检桩 (Module Test Stub)
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("数据加载与增强流水线 - 核心组件冒烟测试")
    print("=" * 60)

    print("\n类别编码空间实例化完毕 (规模限制: 62 类).")

    # 指定预处理产出的结构化测试目录
    test_target_dir = "../data/processed/train"

    if os.path.exists(test_target_dir):
        print(f"\n尝试挂载测试环境目录: {test_target_dir}")

        # 实例化 Dataset 对象以触发内存载入逻辑
        test_ds = CharacterDataset(
            data_dir=test_target_dir,
            transform=get_train_transform(64)
        )

        print(f"\n数据集总样本负载: {len(test_ds)}")

        # 调用分布监测器
        dist_info = test_ds.get_class_distribution()
        print(f"\n频次直方分布数据 (Top 10):")
        for class_id in range(10):
            count_val = dist_info.get(class_id, 0)
            print(f"  编码 {class_id:>2d} [{CLASS_IDX_TO_NAME[class_id]:>1s}] : {count_val:>5d} 样本")

        # 触发渲染诊断组件
        visualize_samples(test_ds, save_path="dataset_augmented_samples.png")
    else:
        print(f"\n[Error]: 依赖路径断裂 -> {test_target_dir}")
        print("执行提示：模块自检依赖前置物理文件，需优先执行 tools/preprocess 生成结构化数据。")

    print("\n" + "=" * 60)
    print("模块自检状态评估完成。")
    print("=" * 60)