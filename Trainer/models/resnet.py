import torch
import torch.nn as nn
from models.layers.resnet_layers import ResBlock
from models.layers.cdcn_layers import Conv2d_cd


class ResNet(nn.Module):

    def __init__(self, layers, im_ch, num_classes, conv_type="conv2d"):
        self.in_channels = 64
        list_conv = ["conv2d", "cdc"]
        if conv_type not in list_conv:
            raise ValueError(f"{conv_type} not in {list_conv}, please choose between two of them")
        base_conv = Conv2d_cd if conv_type == "cdc" else nn.Conv2d
        self.conv1 = base_conv(im_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu= nn.ReLU
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Ini intinya mah layer resnet ada di bawah ini
        res_layers = []

        for i, layer in enumerate(layers):
            stride = 1 if i == 0 else 2
            base_ch = 64
            res_layer = self._make_layer(
                ResBlock, layer, intermediate_channels= base_ch * (i+1), stride=stride
            )
            res_layers.append(res_layer)
        
        self.res_blocks = nn.Sequential(*res_layers)
        self.lastconv = base_conv(2048, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.res_blocks(x)
        outmap = self.lastconv(x)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1) # flatten
        x = self.fc(x)
        outmap = nn.Sigmoid(outmap) # convert into 0-1
        out_score = nn.Sigmoid(x)
        return outmap, out_score
        
    def _make_layer(self, block, base_conv, num_residual_blocks, intermediate_channels, stride):
        identity_downsample= None
        layers= []


        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                base_conv(self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(intermediate_channels * 4)
            )
        
        layers.append(
            block(self.in_channels, intermediate_channels,base_conv, identity_downsample, stride)
        )
        
        # expansion size alwas 4 for ResNet 50, 101, 152

        self.in_channels = intermediate_channels * 4
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))
        
        return nn.Sequential(*layers)

        