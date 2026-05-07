"""模型基础工具函数"""

from typing import Tuple
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """计算模型可训练参数总数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_output_shape(model: nn.Module, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """动态探测模型输出形状"""
    model.eval()
    with torch.no_grad():
        dummy_input = torch.zeros(1, *input_shape)
        output = model(dummy_input)
        return output.shape[1:]
