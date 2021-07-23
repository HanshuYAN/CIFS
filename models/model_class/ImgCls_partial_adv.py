import os, argparse, pathlib
import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from .BaseModel import BaseModelDNN
from .nets.LeNet import LeNet5
from .nets.DnCNN import DnCNN
from .nets.NeuralODE import ODENet

from metric.classification import benchmark_defense_success_rate, dataset_accuracy
from utils import timer, get_epoch_logger
from advertorch.context import ctx_noparamgrad_and_eval


class MNIST_CNN(BaseModelDNN):
    def __init__(self, device='cuda', GPU_IDs=[0], is_train=False) -> None:
        super(MNIST_CNN, self).__init__()
        
        self.net = LeNet5().to(device)
        self.device = device
        self.GPU_IDs = GPU_IDs
        if len(GPU_IDs) > 1:
            self.net = nn.DataParallel(module=self.net, device_ids=GPU_IDs)
        
        if is_train:
            self.optimizer = optim.Adam(self.net.parameters(), lr=1e-4)
            self.optimizer_partial = optim.Adam(itertools.chain(self.net.linear1.parameters(), self.net.linear2.parameters()), lr=1e-4)
            self.criterion = nn.CrossEntropyLoss(reduction="mean").to(device)
            self.scheduler = None
            self.tr_epochs = 100
            self.log_interval = 200
        else:
            self.eval_mode()
            self.set_requires_grad([self.net], False)
    
    def eval_mode(self):
        self.net.eval()
        
    def train_mode(self):
        self.net.train()
        
    def load_networks(self, path):
        self.net.load_state_dict(state_dict=torch.load(path))
    
    def predict(self, x):
        return self.net(x)
    
    def fit(self, train_loader, test_loader,
            logger, save_path='.', 
            is_AdvTr=False, adversary=None, attack=None, attack_kwargs=None):        
        tic_toc = timer()
        epoch_logger = get_epoch_logger()
        for epoch in range(self.tr_epochs):
            logger.info('Training Epoch: {}; Learning rate: {}  .....'.format(epoch, self.optimizer.param_groups[0]['lr']))
            if not is_AdvTr:
                self.train_mode()
                self.fit_one_epoch(train_loader, logger=logger, is_AdvTr=is_AdvTr, adversary=None, device=self.device)
                self.eval_mode()
                accuracy, message = dataset_accuracy(self.predict, test_loader)
            else:
                self.train_mode()
                self.fit_one_epoch(train_loader, logger=logger, is_AdvTr=is_AdvTr, adversary=adversary, device=self.device)
                self.eval_mode()
                accuracy, defense_success_rate, message, _ = benchmark_defense_success_rate(
                    self.predict, test_loader, attack, attack_kwargs, device="cuda") 
                               
            logger.info(message + '\t Time for an epoch: {:.2f}'.format(tic_toc.toc()))

            epoch_logger.epochs.append(epoch)
            epoch_logger.results.append(accuracy)
            best_epoch, message = epoch_logger.update_best_epoch()
            logger.info(message)
            
            model_state_dict = self.net.state_dict() if len(self.GPU_IDs)<2 else self.net.module.state_dict()
            if best_epoch == epoch:
                torch.save(model_state_dict, os.path.join(save_path, "model_best.pt"))
            torch.save(model_state_dict,os.path.join(save_path, 'model_{:03d}.pt'.format(epoch)))
            
    def fit_one_epoch(self, train_loader, logger, is_AdvTr=False, adversary=None, device=torch.device("cuda")):
        tic_toc = timer()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # update all params with clean examples
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            output = self.predict(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            # update deep-partial params with adv examples
            if is_AdvTr:
                self.eval_mode()
                self.set_requires_grad([self.net], False)
                data_adv = adversary.perturb(data, target)
                self.set_requires_grad([self.net], True)
                self.train_mode()

                self.optimizer_partial.zero_grad()
                output = self.predict(data_adv)
                loss_partial = self.criterion(output, target)
                loss_partial.backward()
                self.optimizer_partial.step()
                
                if batch_idx % self.log_interval == 0:
                    logger.info('[{}/{} ({:.0f}%)], Loss: {:.3f},{:.3f}; Time for Batches: {:03f}'.format(
                        batch_idx *len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), 
                        loss.item(), loss_partial.item(),
                        tic_toc.toc()))    
            else:
                if batch_idx % self.log_interval == 0:
                    logger.info('[{}/{} ({:.0f}%)], Loss: {:.6f}, Time for Batches: {:03f}'.format(
                        batch_idx *len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), 
                        loss.item(), 
                        tic_toc.toc()))
        
        if self.scheduler is not None:
            self.scheduler.step()