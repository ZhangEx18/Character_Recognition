import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

        self.downsample = downsample

    def forward(self, x):
        identity = x

        # 如果外部传了下采样模块，就用它处理 identity
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.features(x)
        out += identity
        out = F.relu(out, inplace=True)

        return out

class ResNet(nn.Module):

    def __init__(self,num_classes: int =62):
        super(ResNet,self).__init__()
        self.in_channels = 32





















