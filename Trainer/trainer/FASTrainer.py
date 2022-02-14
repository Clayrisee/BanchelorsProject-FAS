import os
import torch
import torchvision
from trainer.base import BaseTrainer


class FASTrainer(BaseTrainer):
    def __init__(self, cfg, network, optimizer, criterion, dataset, lr_scheduler, device, logger):
        self.cfg = cfg
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataset = dataset
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.logger = logger
    
    def load_model(self):
        pass
    
    def save_model(self):
        pass
    
    def train_one_epoch(self):
        pass
    
    def train(self):
        pass
    
    def validate(self):
        pass