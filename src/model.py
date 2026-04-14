import torch
import torch.nn as nn
import torch.nn.functional as f
#import Tuple


class ResidualBlock(nn.Module):
	def __init__(self,in_channles:int,out_channles:int,stride:int = 1):
		super(ResidualBlock,self).__init__()
		self.conv1 = nn.Conv2d(in_channles,out_channles,kernel_size=3,stride=stride,padding=1)
		self.bn1 = nn.BatchNorm2d(out_channles)
		self.conv2 = nn.Conv2d(in_channles,out_channles,kernel_size=3,stride=1,padding=1)
		self.bn2 = nn.BatchNorm2d(out_channles)

		self.shortcut = nn.Sequential()
		if stride  != 1 or in_channles != out_channles:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channles,out_channles,kernel_size=1,stride=1),
				nn.BatchNorm2d(out_channles)
			)

	def forward(self,x:torch.Tensor) -> torch.Tensor:
		out = f.relu(self.bn1(self.conv1(x)))

		out = self.bn2(self.conv2(x))
		out += self.shortcut(x)
		out = f.relu(out)

		return out

