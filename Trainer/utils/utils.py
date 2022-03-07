import yaml
import torch
from torch import optim
from models.models import *

def read_cfg(cfg_file):
    """
    Read configurations from yaml file
    Args:
        cfg_file (.yaml): path to cfg yaml
    Returns:
        (dict): configuration in dict
    """
    with open(cfg_file, 'r') as rf:
        cfg = yaml.safe_load(rf)
        return cfg

@torch.no_grad()
def scoring_method(pred_mask, pred_score, scoring_method="combine"):
    scoring_method_list = ["combine", "label", "avg_mask"]
    final_score = 0
    
    if scoring_method not in scoring_method_list:
        raise NotImplementedError
    
    if scoring_method == "combine":
        final_score = (torch.mean(pred_mask, axis=(1, 2)) + pred_score) / 2

    elif scoring_method == "label":
        final_score = pred_score

    else:
        final_score = torch.mean(pred_mask, axis=(1, 2))

    return final_score

def export_model_to_onnx(model, dataset, output_path):
    example_input = next(iter(dataset.train_dataloader()))[0]
    example_input = example_input[:1]
    print(example_input.shape)
    print(type(example_input))
    # print()
    model.to_onnx(output_path, example_input, export_params = True)

def get_optimizer(cfg, network):
    """ Get optimizer based on the configuration
    Args:
        cfg (dict): a dict of configuration
        network: network to optimize
    Returns:
        optimizer 
    """
    optimizer = None
    if cfg['train']['optimizer'] == 'adam':
        optimizer = optim.Adam(network.parameters(), lr=cfg['train']['lr'])

    elif cfg['train']['optimizer'] == 'adamw':
        optimizer = optim.AdamW(network.parameters(), lr=cfg['train']['lr'])

    elif cfg['train']['optimizer'] == 'sgd':
        optimizer = optim.SGD(network.parameters(), lr=cfg['train']['lr'])

    elif cfg['train']['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(network.parameters(), lr=cfg['train']['lr']) 

    else:
        raise NotImplementedError

    return optimizer

def get_device(cfg):
    """ Get device based on configuration
    Args: 
        cfg (dict): a dict of configuration
    Returns:
        torch.device
    """
    device = None
    if cfg['device'] == '':
        device = torch.device("cpu")
    elif cfg['device'] == '0':
        device = torch.device("cuda:0")
    elif cfg['device'] == '1':
        device = torch.device("cuda:1")
    else:
        raise NotImplementedError
    return device

def build_network(cfg):
    """ Build the network based on the cfg
    Args:
        cfg (dict): a dict of configuration
    Returns:
        network (nn.Module) 
    """
    network = None
    pretrained = cfg['model']['pretrained']
    kwargs = {
        'num_classes': cfg['model']['num_classes'],
    }
    if cfg['model']['base'] == 'convnext_tiny':
        network = convnext_tiny(pretrained **kwargs)
        
    elif cfg['model']['base'] == 'convnext_small':
        network = convnext_tiny(pretrained **kwargs)

    elif cfg['model']['base'] == 'convnext_base':
        network = convnext_tiny(pretrained **kwargs)

    elif cfg['model']['base'] == 'convnext_large':
        network = convnext_tiny(pretrained **kwargs)

    elif cfg['model']['base'] == 'cd_convnext_tiny':
        network = cd_convnext_tiny(**kwargs)

    elif cfg['model']['base'] == 'cd_convnext_small':
        network = cd_convnext_small(**kwargs)

    elif cfg['model']['base'] == 'cd_convnext_base':
        network = cd_convnext_base(**kwargs)

    elif cfg['model']['base'] == 'cd_convnext_large':
        network = cd_convnext_large(**kwargs)
    
    elif cfg['model']['base'] == 'resnet50':
        network = resnet_50(img_ch=3, **kwargs)

    elif cfg['model']['base'] == 'resnet101':
        network = resnet_101(img_ch=3, **kwargs)

    elif cfg['model']['base'] == 'resnet152':
        network = resnet_152(img_ch=3, **kwargs)

    elif cfg['model']['base'] == 'cd_resnet50':
        network = cd_resnet_50(img_ch=3, **kwargs)

    elif cfg['model']['base'] == 'cd_resnet101':
        network = cd_resnet_101(img_ch=3, **kwargs)

    elif cfg['model']['base'] == 'cd_resnet152':
        network = cd_resnet_152(img_ch=3, **kwargs)
    
    elif cfg['model']['base'] == 'cdcn':
        network = cdcn()
    else:
        raise NotImplementedError

    return network
