import os
import torch
from trainer.base import BaseTrainer
from utils.metrics import calculate_liveness_metric
from utils.utils import scoring_method


class FASTrainer(BaseTrainer):
    def __init__(self, cfg, network, optimizer, criterion, dataset, device, lr_scheduler=None, logger=None):
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
        loss = 0
        for i, (imgs, masks, labels) in enumerate(self.trainloader):
            imgs, masks, labels = imgs.to(self.device), masks.to(self.device), labels.to(self.device)
            pred_mask, preds = self.network(imgs)
            self.optimizer.zero_grad()
            loss = self.criterion(pred_mask, preds, masks, labels)
            loss.backward()
            self.optimizer.step()
            loss += loss.item() * imgs.shape[0]
            # TODO: add logger to record loss for each step.

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        
        epoch_loss = loss / len(self.dataset.train_set)
        if self.logger is not None:
            self.logger.log_metric("train_loss", epoch_loss, epoch=epoch)

        return epoch_loss
    
    def train(self):
        for epoch in range(self.cfg['train']['num_epochs']):
            print("="*80)
            print("Epoch: {}".format(epoch))
            train_loss = self.train_one_epoch(epoch)
            val_apcer, val_npcer, val_acer, val_loss = self.validate_one_epoch(epoch)
            print("Train Loss: {:.4f}, Val Loss: {:.4f}\n APCER: {:.4f}, NPCER: {:.4f}, ACER: {:.4f}"\
                .format(train_loss,val_loss, val_apcer, val_npcer, val_acer))
            print("="*80)
    
    def validate_one_epoch(self, epoch):
        apcer = npcer = acer = val_loss = 0
        self.network.eval()
        with torch.no_grad():
            for i, (imgs, masks, labels) in enumerate(self.valloader):
            # # TODO: add scoring method
                imgs, masks, labels = imgs.to(self.device), masks.to(self.device), labels.to(self.device)
                pred_mask, preds = self.network(imgs)
                loss = self.criterion(pred_mask, preds, masks, labels)
                pred_scores = scoring_method(pred_mask, preds)
                liveness_metrics = self.eval_metrics(pred_scores, labels)
                val_loss += loss.item() * imgs.shape[0]
                apcer += liveness_metrics['apcer']
                npcer += liveness_metrics['npcer']
                acer += liveness_metrics['acer']
                # TODO: add logger to record result metric for each step.
            epoch_val_loss = loss / len(self.dataset.val_set)
            epoch_apcer = apcer / len(self.dataset.val_set)
            epoch_npcer = npcer / len(self.dataset.val_set)
            epoch_acer = acer / len(self.dataset.val_set)
            
            if self.logger is not None:
                self.logger.log_metric("epoch_val_loss", epoch_val_loss, epoch=epoch)
                self.logger.log_metric("epoch_apcer", epoch_apcer, epoch=epoch)
                self.logger.log_metric("epoch_npcer", epoch_npcer, epoch=epoch)
                self.logger.log_metric("epoch_acer", epoch_acer, epoch=epoch)
        return epoch_apcer, epoch_npcer, epoch_acer, epoch_val_loss
