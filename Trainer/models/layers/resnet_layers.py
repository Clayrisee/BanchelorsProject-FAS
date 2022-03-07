import torch
import torch.nn as nn
from models.layers.cdcn_layers import Conv2d_cd

class ResBlock(nn.Module):
    def __init__(self,
    in_channels, intermediate_channels, base_conv="conv2d", identity_downsample=None, stride=1
    ):
        super(ResBlock, self).__init__()
        list_conv = ["conv2d", "cdc"]
        if base_conv not in list_conv:
            raise ValueError(f"{base_conv} not in {list_conv}, please choose between two of them")
        
        conv = Conv2d_cd if base_conv == "cdc" else nn.Conv2d
        self.expansion = 4
        self.conv1 = conv(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = conv(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride,
        padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = conv(
                    intermediate_channels,
                    intermediate_channels * self.expansion,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x += identity
        x = self.relu(x)
        return x
