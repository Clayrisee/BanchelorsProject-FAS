import os
import torch
import torchvision
from trainer.base import BaseTrainer
from utils.metrics import calculate_liveness_metric
from utils.utils import scoring_method


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
        self.eval_metrics = calculate_liveness_metric
    
    def load_model(self):
        saved_name = os.path.join(self.cfg['output_dir'], '{}_{}.pth'.format(self.cfg['model']['base'], self.cfg['dataset']['name']))
        state = torch.load(saved_name)
        self.optimizer.load_state_dict(state['optimizer'])
        self.network.load_state_dict(state['state_dict'])
    
    def save_model(self, epoch):
        if not os.path.exists(self.cfg['output_dir']):
            os.makedirs(self.cfg['output_dir'])

        saved_name = os.path.join(self.cfg['output_dir'], '{}_{}.pth'.format(self.cfg['model']['base'], self.cfg['dataset']['name']))

        state = {
            'epoch': epoch,
            'state_dict': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        
        torch.save(state, saved_name)
    
    def train_one_epoch(self, epoch):
        self.network.train()
        self.train_loss_metric.reset(epoch)
        self.train_acc_metric.reset(epoch)

        for i, (imgs, masks, labels) in enumerate(self.trainloader):
            imgs, masks, labels = imgs.to(self.device), masks.to(self.device), labels.to(self.device)
            pred_mask, preds = self.network(imgs)
            self.optimizer.zero_grad()
            loss = self.criterion(pred_mask, preds, masks, labels)
            loss.backward()
            self.optimizer.step()



            # Update metrics
            # self.train_loss_metric.update(loss.item())
            # self.train_acc_metric.update(accuracy)

            # print('Epoch: {}, iter: {}, loss: {}, acc: {}'.format(epoch, epoch * len(self.trainloader) + i, self.train_loss_metric.avg, self.train_acc_metric.avg))

    
    def train(self):
        pass
    
    def validate(self):
        pass
            # # TODO: add scoring method
        # pred_scores = scoring_method(pred_mask, preds)
        # liveness_metrics = self.eval_metrics(pred_scores, labels)