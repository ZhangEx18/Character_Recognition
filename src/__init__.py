"""
CNN 字符识别项目 - 核心算法包 (src)

本项目算法核心枢纽。提供多层级模型架构（Simple/Detailed/ResNet）、
数据流水线、实时推理引擎及跨平台硬件调度工具。
通过此文件统一对外暴露核心接口，彻底隔离并隐藏内部复杂的相对路径结构。
"""

__version__ = "1.0.0"

# ============================================================
# 核心组件导出区
# 采用相对导入 (".") 将内部各子文件的类与方法挂载至顶级命名空间
# ============================================================

# 1. 模型架构库：提供从快速验证到深层残差的三阶模型引擎及参数分析工具
from .model import DetailedCNN, SimpleCNN, ResNet, SEResNet, FocalLoss, count_parameters

# 2. 数据与映射字典：负责物理文件调度、预处理增强及全局类别常量 (共 62 类)
from .dataset import (
    create_dataloaders,
    CharacterDataset,
    CLASS_IDX_TO_NAME,
    CLASS_NAME_TO_IDX,
    NUM_CLASSES
)

# 3. 推理引擎：面向对象封装的黑盒预测器，支持单张/批量高效推断
from .inference import Predictor

# 4. 硬件控制层：自动嗅探并调度最强算力 (Mac MPS / CUDA / CPU)
from .utils import get_device

# ============================================================
# 模块导出白名单 (最佳工程实践)
# 严格限制外部通过 `from src import *` 导入时所能获取的内容。
# 仅暴露以下核心 API，防止内部临时变量或第三方库污染调用者的命名空间。
# ============================================================
__all__ = [
    # 模型相关
    "DetailedCNN",
    "SimpleCNN",
    "ResNet",
    "SEResNet",
    "FocalLoss",
    "count_parameters",

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