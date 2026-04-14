import torch
import torch.nn as nn
import torch.nn.functional as f
#import Tuple


class ResidualBlock(nn.Module):
	def __init__(self,in_channles:int,out_channles:int,stride:int = 1):
		super(ResidualBlock,self).__init__()
		self.conv1 = nn.Conv2d(in_channles,out_channles,kernel_size=3,stride=stride)
		self.bn1 = nn.BatchNorm2d(out_channles)