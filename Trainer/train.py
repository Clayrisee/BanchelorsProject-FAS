import os
from comet_ml import Experiment
from data.dataloader import LivenessDataModule
from utils.utils import read_cfg, get_optimizer, get_device, build_network
from trainer.FASTrainer import FASTrainer
from utils.loss import PixFocalLoss, PixWiseBCELoss
from utils.schedulers import CosineAnealingWithWarmUp

cfg = read_cfg(cfg_file="config/base_config.yaml")
logger = Experiment(api_key=cfg['logger']['api_key'], 
        project_name=cfg['logger']['project_name'],
        workspace=cfg['logger']['workspace'])
device = get_device(cfg)
network = build_network(cfg)
optimizer = get_optimizer(cfg, network)
lr_scheduler = CosineAnealingWithWarmUp(optimizer, 
    first_cycle_steps=300, 
    cycle_mult=1.0,
    max_lr=0.1, 
    min_lr=0.001, 
    warmup_steps=50, 
    gamma=0.5)

criterion = PixFocalLoss()

dataset = LivenessDataModule(cfg)

trainer = FASTrainer(cfg, network, optimizer, criterion, dataset, device, lr_scheduler, logger=logger)

trainer.train()
