"""
CNN 字符识别项目 - 核心算法包 (src)

提供多层级模型架构、数据流水线、实时推理引擎及跨平台硬件调度工具。
"""

__version__ = "2.0.0"

# 模型架构库
from .models import (
    SimpleCNN,
    DetailedCNN,
    ResNet,
    SEResNet,
    MLPNet,
    LeNet,
    VGG,
    MobileNet,
    count_parameters,
    create_model,
    list_models,
    MODEL_REGISTRY,
)

# 数据与映射字典
from .dataset import (
    create_dataloaders,
    CharacterDataset,
    CLASS_IDX_TO_NAME,
    CLASS_NAME_TO_IDX,
    NUM_CLASSES,
)

# 推理引擎
from .inference import Predictor

# 硬件控制层
from .utils import get_device

# 保留旧版 FocalLoss 导出（在 models 中未拆分，仍从 model.py 导入）
from .model import FocalLoss

__all__ = [
    # 模型相关
    "SimpleCNN",
    "DetailedCNN",
    "ResNet",
    "SEResNet",
    "MLPNet",
    "LeNet",
    "VGG",
    "MobileNet",
    "FocalLoss",
    "count_parameters",
    "create_model",
    "list_models",
    "MODEL_REGISTRY",

    # 数据集相关
    "create_dataloaders",
    "CharacterDataset",
    "CLASS_IDX_TO_NAME",
    "CLASS_NAME_TO_IDX",
    "NUM_CLASSES",

    # 业务功能相关
    "Predictor",
    "get_device",
]
