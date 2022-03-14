from models.layers.convnext_layers import LayerNorm, Block
from models.layers.cdcn_layers import Conv2d_cd
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1., conv_type="conv2d",
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.base_conv = Conv2d_cd if conv_type=="cdc" else nn.Conv2d
        
        stem = nn.Sequential(
            self.base_conv(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i],base_conv=self.base_conv, drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i],)],
            
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)
        self.dec = nn.Conv2d(dims[-1], 1, kernel_size=1, stride=1) # define dec based on last dim
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (self.base_conv, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        embedding = self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)
        return  x, embedding

    def forward(self, x):
        outmap, embedding = self.forward_features(x) # forward feature to feature extraction
        # # TODO: Fix this line. Kayaknya salah define layer. coba dibenerin dlu dah
        # print(outmap.shape)
        # in_c, out_c = (outmap.shape[1], 1) # get input ch and output ch for decryptor
        
        # self.dec = self.dec(in_c, out_c) # define decryptor layer
        outmap = self.dec(outmap) # return outmap for pixelwise supervision
        x = self.head(embedding) # MLP for classifier
        x = torch.flatten(x) # return final label
        x = x.unsqueeze(1)
        outmap, x = F.sigmoid(outmap), F.sigmoid(x)
        # print('outmap',outmap)
        # print('x',x)
        return outmap, x