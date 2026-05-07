"""神经网络模型库 - 字符识别核心架构

统一适配 64×64 灰度图、62 类字符识别任务的经典神经网络：
- MLPNet:    全连接网络，无卷积，baseline 对比
- LeNet:     CNN 始祖（1998），2 层卷积 + 3 层全连接
- SimpleCNN: 浅层 2 块卷积，参数量小，快速验证
- DetailedCNN: 3 块卷积 + BN + Dropout，生产推荐
- VGG:       VGG 风格，3×3 小卷积核堆叠
- ResNet:    残差网络，跳跃连接解决梯度消失
- SEResNet:  SE-ResNet，通道注意力增强
- MobileNet: 深度可分离卷积，轻量高效
"""

from .simple_cnn import SimpleCNN
from .detailed_cnn import DetailedCNN
from .resnet import ResNet
from .seresnet import SEResNet
from .mlp_net import MLPNet
from .lenet import LeNet
from .vgg import VGG
from .mobilenet import MobileNet
from .base import count_parameters, get_output_shape

# 模型注册表：名称 -> (类, 描述)
MODEL_REGISTRY = {
    "mlp": (MLPNet, "全连接 MLP，无卷积，baseline 对比"),
    "lenet": (LeNet, "LeNet-5，CNN 始祖，1998"),
    "simple": (SimpleCNN, "浅层 CNN，参数量最小，快速验证"),
    "detailed": (DetailedCNN, "带 BN/Dropout 的标准 CNN，生产推荐"),
    "vgg": (VGG, "VGG 风格，3×3 小卷积核堆叠"),
    "resnet": (ResNet, "残差网络，4-stage 深层特征提取"),
    "seresnet": (SEResNet, "SE-ResNet，通道注意力增强"),
    "mobilenet": (MobileNet, "MobileNet，深度可分离卷积，轻量"),
}

MODEL_ALIASES = {
    "simplecnn": "simple",
    "detailedcnn": "detailed",
    "seres": "seresnet",
    "fcn": "mlp",
    "fc": "mlp",
}


def create_model(net_type: str, num_classes: int = 62):
    """根据名称创建模型实例"""
    net_type = net_type.lower().strip()
    net_type = MODEL_ALIASES.get(net_type, net_type)

    if net_type not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"不支持的模型类型: {net_type}。可用选项: {available}")

    model_cls, _ = MODEL_REGISTRY[net_type]
    return model_cls(num_classes=num_classes)


def list_models():
    """返回所有可用模型的名称和描述"""
    return {name: desc for name, (_, desc) in MODEL_REGISTRY.items()}


__all__ = [
    "SimpleCNN",
    "DetailedCNN",
    "ResNet",
    "SEResNet",
    "MLPNet",
    "LeNet",
    "VGG",
    "MobileNet",
    "count_parameters",
    "get_output_shape",
    "create_model",
    "list_models",
    "MODEL_REGISTRY",
]
