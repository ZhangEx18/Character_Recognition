"""前向传播可视化 - 通过 hooks 捕获每层特征图

使用 register_forward_hook 拦截模型各层的输出，
将特征图转换为缩略图网格，用于前端动画展示。
"""

import io
import base64
from typing import Dict, List

import torch
import torch.nn as nn
import numpy as np
from PIL import Image


class ForwardVisualizer:
    """前向传播可视化器

    用法:
        viz = ForwardVisualizer(model)
        viz.capture(input_tensor)
        images = viz.to_images()
        viz.remove_hooks()
    """

    def __init__(self, model: nn.Module, layer_names: List[str] = None):
        self.model = model
        self.features: Dict[str, torch.Tensor] = {}
        self.hooks: List = []

        if layer_names is None:
            layer_names = self._auto_discover_layers(model)

        for name in layer_names:
            layer = self._get_layer(model, name)
            if layer is not None:
                hook = layer.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def _auto_discover_layers(self, model: nn.Module) -> List[str]:
        """自动发现关键可视化层（限制数量）"""
        candidates = []
        for name, module in model.named_modules():
            # 只选 Conv2d 和关键 Linear 层
            if isinstance(module, nn.Conv2d):
                candidates.append(name)
            elif isinstance(module, nn.Linear) and 'fc' in name.lower():
                candidates.append(name)

        # 限制最多 8 个层，均匀采样
        if len(candidates) > 8:
            step = len(candidates) // 8
            candidates = candidates[::step][:8]

        return candidates

    def _get_layer(self, model: nn.Module, name: str):
        """通过点分隔路径获取子模块"""
        parts = name.split('.')
        module = model
        for part in parts:
            if part == '':
                continue
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part, None)
            if module is None:
                return None
        return module

    def _make_hook(self, name: str):
        def hook(_module, _input, output):
            self.features[name] = output.detach().cpu()
        return hook

    def capture(self, x: torch.Tensor):
        """执行前向传播并捕获所有 hooked 层的输出"""
        self.features = {}
        self.model.eval()
        with torch.no_grad():
            self.model(x)
        return self.features

    def to_images(self, max_channels: int = 16, thumb_size: int = 64) -> Dict[str, str]:
        """将捕获的特征图转为 base64 PNG

        返回: {layer_name: base64_png_string}
        """
        images = {}
        for name, feature in self.features.items():
            if feature.dim() == 4:  # [B, C, H, W]
                img = self._conv_feature_to_image(feature[0], max_channels, thumb_size)
            elif feature.dim() == 2:  # [B, C]
                img = self._fc_feature_to_image(feature[0])
            else:
                continue

            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            # 简化层名用于展示
            images[self._simplify_name(name)] = img_b64

        return images

    def _simplify_name(self, name: str) -> str:
        """简化层名用于前端展示"""
        name = name.replace('layer', 'L').replace('conv', 'Conv')
        name = name.replace('bn', 'BN').replace('fc', 'FC')
        # 截断过长名称
        if len(name) > 20:
            parts = name.split('.')
            if len(parts) > 2:
                name = '.'.join(parts[-2:])
        return name

    def _conv_feature_to_image(self, feature: torch.Tensor,
                               max_channels: int, thumb_size: int) -> Image.Image:
        """将 [C, H, W] 卷积特征图转为缩略图网格"""
        C = min(feature.shape[0], max_channels)
        grid_n = int(np.ceil(np.sqrt(C)))

        # 归一化到 [0, 255]
        f = feature[:C]
        f_min, f_max = f.min(), f.max()
        if f_max > f_min:
            f = (f - f_min) / (f_max - f_min) * 255
        else:
            f = f * 0
        f = f.numpy().astype(np.uint8)

        # 如果特征图太大，先缩小
        h, w = f.shape[1], f.shape[2]
        if h > thumb_size or w > thumb_size:
            # 使用 PIL 批量缩放
            f_list = [Image.fromarray(f[i]) for i in range(C)]
            f_list = [img.resize((thumb_size, thumb_size), Image.BILINEAR) for img in f_list]
            f = np.stack([np.array(img) for img in f_list])
            h = w = thumb_size

        # 组装网格
        grid = np.zeros((grid_n * h, grid_n * w), dtype=np.uint8)
        for i in range(C):
            row, col = i // grid_n, i % grid_n
            grid[row*h:(row+1)*h, col*w:(col+1)*w] = f[i]

        return Image.fromarray(grid)

    def _fc_feature_to_image(self, feature: torch.Tensor) -> Image.Image:
        """将 FC 输出转为热力条图像"""
        values = feature.numpy()
        n = len(values)

        # 归一化到 [-1, 1] 用于颜色映射
        v_min, v_max = values.min(), values.max()
        if v_max > v_min:
            norm = (values - v_min) / (v_max - v_min)
        else:
            norm = np.zeros_like(values)

        # 创建热力图：蓝色(低) -> 白色(中) -> 红色(高)
        h, w = 32, max(256, n * 4)
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # 每个值画一个色块
        block_w = w // n
        for i, v in enumerate(norm):
            x0 = i * block_w
            x1 = min((i + 1) * block_w, w)
            if v < 0.5:
                # 蓝色到白色
                r = int(v * 2 * 255)
                g = int(v * 2 * 255)
                b = 255
            else:
                # 白色到红色
                r = 255
                g = int((1 - v) * 2 * 255)
                b = int((1 - v) * 2 * 255)
            img[:, x0:x1] = [r, g, b]

        return Image.fromarray(img)

    def remove_hooks(self):
        """清理所有注册的 hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hooks()
