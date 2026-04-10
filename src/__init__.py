"""
CNN 字符识别项目 - 核心算法包 (src)

提供模型架构、数据预处理流水线、实时推理引擎及基础辅助工具。
对外提供统一的调用接口，隐藏内部复杂的相对路径。
"""

__version__ = "1.0.0"

# ==========================================
# 从内部各个子文件中，把最核心的工具“端”到前台来
# 注意前面的 "." 表示从当前 src 目录查找
# ==========================================

# 1. 模型架构类
from .model import DetailedCNN, SimpleCNN, count_parameters

# 2. 数据处理与映射类
from .dataset import (
    create_dataloaders,
    CharacterDataset,
    CLASS_IDX_TO_NAME,
    CLASS_NAME_TO_IDX
)

# 3. 推理引擎类
from .inference import Predictor

# 4. 硬件底层工具
from .utils import get_device

# ==========================================
# 白名单清单 (最佳实践)
# 告诉 Python：如果有外部脚本偷懒写了 `from src import *`，
# 那么系统只会把下面列表里的东西交出去，防止变量污染。
# ==========================================
__all__ = [
    "DetailedCNN",
    "SimpleCNN",
    "count_parameters",
    "create_dataloaders",
    "CharacterDataset",
    "CLASS_IDX_TO_NAME",
    "CLASS_NAME_TO_IDX",
    "Predictor",
    "get_device",
]