import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):  # 适用于ResNet-18，34
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            # 下面这行的 kernel_size 应该改为 3，并且加上 padding=1
            nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel * self.expansion)  # 严谨起见，保持维度名称一致
        )

        self.downsample = downsample

    def forward(self, x):
        identity = x

        # 如果外部传了下采样模块，就用它处理 identity
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.features(x)
        out += identity
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):  # 适用于ResNet-50，101
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super().__init__()
        self.features = nn.Sequential(
            # 第一层：1x1 卷积，降维
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),  # 记得这里通常有激活函数

            # 第二层：3x3 卷积，处理特征（必须有 padding=1，且 stride 跟随输入）
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),

            # 第三层：1x1 卷积，升维 (乘以 expansion 4)
            nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel * self.expansion),
            # 注意：最后这一层后面没有 ReLU，ReLU 要在残差相加之后做
        )
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.features(x)
        out += identity  # 相加
        out = F.relu(out)  # 相加后再激活

        return out


class ResNet(nn.Module):

    def __init__(self, block, block_num, num_classes: int = 62, include_top=True):
        super().__init__()
        self.in_channels = 64
        self.include_top = include_top

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, bias=False, padding=3)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None  # 1. 必须先初始化为 None

        if stride != 1 or self.in_channels != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []
        # 2. 传入第一个 block，注意补上逗号
        layers.append(block(
            self.in_channels,
            channel,
            stride=stride,
            downsample=downsample
        ))

        # 更新全局的通道数，供后面的 block 使用
        self.in_channels = channel * block.expansion

        # 3. 正确实例化后续的 block 并添加进列表
        for _ in range(1, block_num):
            # 后续的 block 不需要下采样，stride 默认为 1
            layers.append(block(self.in_channels, channel))

        return nn.Sequential(*layers)