import torch
from models.convnext import ConvNeXt
from models.resnet import ResNet
from models.cdcn import CDCN


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

# Function to Generate ConvNext with Vanilla Convolution
def convnext_tiny(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], conv_type="conv2d",**kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


def convnext_small(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], conv_type="conv2d",**kwargs)
    if pretrained:
        url = model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url,map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

def convnext_base(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], conv_type="conv2d", **kwargs)
    if pretrained:
        url = model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

def convnext_large(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], conv_type="conv2d", **kwargs)
    if pretrained:
        url = model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

# Function to Generate ConvNext with Central Difference Convolution

def cd_convnext_tiny(**kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], 
            dims=[96, 192, 384, 768], 
            conv_type="cdc",
            **kwargs)
    return model


def cd_convnext_small(**kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], 
            dims=[96, 192, 384, 768], 
            conv_type="cdc", 
            **kwargs)
    return model

def cd_convnext_base(**kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], 
            dims=[128, 256, 512, 1024], 
            conv_type="cdc", 
            **kwargs)
    return model

def cd_convnext_large(**kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], 
            dims=[192, 384, 768, 1536], 
            conv_type="cdc", 
            **kwargs)
    return model


# Function to Generate ResNet models using Vanilla Convolution

def resnet_50(img_ch=3, **kwargs):
    model = ResNet([3, 4, 6, 3], img_ch, **kwargs)
    return model

def resnet_101(img_ch=3, **kwargs):
    model = ResNet([3, 4, 23, 3], img_ch, **kwargs)
    return model

def resnet_152(img_ch=3, **kwargs):
    model = ResNet([3, 8, 36, 3], img_ch, **kwargs)
    return model

# Function to Generate ResNet model using Central Difference Convolution

def cd_resnet_50(img_ch=3, **kwargs):
    model = ResNet([3, 4, 6, 3], img_ch,conv_type="cdc" **kwargs)
    return model

def cd_resnet_101(img_ch=3, **kwargs):
    model = ResNet([3, 4, 23, 3], img_ch,conv_type="cdc" **kwargs)
    return model

def cd_resnet_152(img_ch=3, **kwargs):
    model = ResNet([3, 8, 36, 3], img_ch,conv_type="cdc" **kwargs)
    return model

# Function to generate CDCN
def cdcn():
    return CDCN()