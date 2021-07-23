import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import numpy as np

from .BaseModel import BaseModelDNN

from utils import timer, get_epoch_logger
from datasets.utils import dtype_mean_shift, np2Tensor, tensor2im

from metric.imgprocessing import dataset_performance_evaluator

from models.nets.DnCNN import DnCNN
from models.nets.NeuralODE import ODENet
    
class DnNODE(BaseModelDNN):
    def __init__(self, device='cuda', GPU_IDs=[0], is_train=False) -> None:
        super().__init__()
        
        self.net = ODENet().to(device)
        self.device = device
        self.GPU_IDs = GPU_IDs
        if len(GPU_IDs) > 1:
            self.net = nn.DataParallel(module=self.net, device_ids=GPU_IDs)
            
        if is_train:
            self.optimizer = optim.Adam(self.net.parameters(), lr=0.01, betas=(0.5, 0.999),eps=1e-8) # 0.001
            self.criterion = nn.MSELoss(reduction='mean').to(device)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.5)
            self.tr_epochs = 300
            self.log_interval = 50
    
    def eval_mode(self):
        self.net.eval()
        
    def train_mode(self):
        self.net.train()
        
    def load_networks(self, path):
        self.net.load_state_dict(state_dict=torch.load(path))
    
    def predict(self, x):
        return self.net(x)
    
    def predict_numpy_image(self, img_n):
        # Input an numpy image (h ,w, 3/1); Output an numpy image with size unchanged
        img_n = dtype_mean_shift([np.expand_dims(img_n, axis=2)], rgb_range=1, rgb_mean=[0], rgb_std=[1], mode='-')[0]
        img_n = np2Tensor([img_n])[0]
        img_n = torch.reshape(img_n, (-1, img_n.shape[0], img_n.shape[1], img_n.shape[2]))
        img_n = img_n.to(self.device)
        img_n_dn = self.net(img_n)           
        img_n_dn = tensor2im(img_n_dn, 1, [0], [1])
        return img_n_dn

    
    def fit(self, train_loader, logger, save_path='.'):
        tic_toc = timer(); epoch_logger = get_epoch_logger()

        for epoch in range(self.tr_epochs):            
            logger.info('Training Epoch: {}; Learning rate: {}  .....'.format(epoch, self.optimizer.param_groups[0]['lr']))

            self.train_mode()
            self.fit_one_epoch(train_loader, logger)

            self.eval_mode()
            psnr, message = dataset_performance_evaluator(self.predict_numpy_image, dir_clean='./datasets/.benchmarks/BSDS300/images/test', ext='.jpg', n_colors=1, noise=['G', 15/255], 
                                                dir_saving='.', save_results=False)
            logger.info(message + '\t Time for an epoch: {:.2f}'.format(tic_toc.toc()))

            epoch_logger.epochs.append(epoch)
            epoch_logger.results.append(psnr)
            best_epoch, message = epoch_logger.update_best_epoch()
            logger.info(message)
            
            model_state_dict = self.net.state_dict() if len(self.GPU_IDs)<2 else self.net.module.state_dict()
            if best_epoch == epoch:
                torch.save(model_state_dict, os.path.join(save_path, "model_best.pt"))
            torch.save(model_state_dict,os.path.join(save_path, 'model_{:03d}.pt'.format(epoch)))


    def fit_one_epoch(self, train_loader, logger):
        tic_toc = timer()
        
        for batch_idx, (data, target, path) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.predict(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            if batch_idx % self.log_interval == 0:
                logger.info('[{}/{} ({:.0f}%)], Loss: {:.6f}, Time for Batches: {:03f}'.format(
                    batch_idx *len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), 
                    loss.item(), 
                    tic_toc.toc()))
        self.scheduler.step()



class DnCNN400(BaseModelDNN):
    def __init__(self, device='cuda', GPU_IDs=[0], is_train=False) -> None:
        super(DnCNN400, self).__init__()
        
        self.net = DnCNN(depth=8).to(device)
        self.device = device
        self.GPU_IDs = GPU_IDs
        if len(GPU_IDs) > 1:
            self.net = nn.DataParallel(module=self.net, device_ids=GPU_IDs)
            
        if is_train:
            self.optimizer = optim.Adam(self.net.parameters(), lr=0.001, betas=(0.5, 0.999),eps=1e-8)
            self.criterion = nn.MSELoss(reduction='mean').to(device)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
            self.tr_epochs = 300
            self.log_interval = 50
    
    def eval_mode(self):
        self.net.eval()
        
    def train_mode(self):
        self.net.train()
    
    def load_networks(self, path):
        self.net.load_state_dict(state_dict=torch.load(path))
        
    def predict(self, x):
        return self.net(x)
    
    def predict_numpy_image(self, img_n):
        # Input an numpy image (h ,w, 3/1); Output an numpy image with size unchanged
        img_n = dtype_mean_shift([np.expand_dims(img_n, axis=2)], rgb_range=1, rgb_mean=[0], rgb_std=[1], mode='-')[0]
        img_n = np2Tensor([img_n])[0]
        img_n = torch.reshape(img_n, (-1, img_n.shape[0], img_n.shape[1], img_n.shape[2]))
        img_n = img_n.to(self.device)
        img_n_dn = self.net(img_n)           
        img_n_dn = tensor2im(img_n_dn, 1, [0], [1])
        return img_n_dn

    
    def fit(self, train_loader, logger, save_path='.'):
    
        tic_toc = timer()
        epoch_logger = get_epoch_logger()
        
        for epoch in range(self.tr_epochs):
            
            logger.info('Training Epoch: {}; Learning rate: {}  .....'.format(epoch, self.optimizer.param_groups[0]['lr']))

            fit_one_epoch(self.net, train_loader, 
                          self.criterion, self.optimizer, self.scheduler, 
                          tic_toc, logger, self.log_interval, device=self.device)

            self.eval_mode()
            psnr, message = dataset_performance_evaluator(self.predict_numpy_image, dir_clean='./datasets/.benchmarks/BSDS300/images/test', ext='.jpg', n_colors=1, noise=['G', 15/255], 
                                                dir_saving='.', save_results=False)
            logger.info(message)

            epoch_logger.epochs.append(epoch)
            epoch_logger.results.append(psnr)
            best_epoch, message = epoch_logger.update_best_epoch()
            logger.info(message)
            
            model_state_dict = self.net.state_dict() if len(self.GPU_IDs)<2 else self.net.module.state_dict()
            if best_epoch == epoch:
                torch.save(model_state_dict, os.path.join(save_path, "model_best.pt"))
            torch.save(model_state_dict,os.path.join(save_path, 'model_{:03d}.pt'.format(epoch)))


def fit_one_epoch(net, train_loader, criterion, optimizer, scheduler, timer, logger, log_interval, device):
    timer.tic()
    net.train()
    
    for batch_idx, (data, target, path) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            logger.info('[{}/{} ({:.0f}%)], Loss: {:.6f}, Time for Batches: {:03f}'.format(
                batch_idx *len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), 
                loss.item(), 
                timer.toc()))
            
    scheduler.step()



if __name__ == '__main__':
    from .nets.DnCNN import DnCNN
    model = DnCNN(depth=6)
    input = torch.rand((1, 1, 100, 100))
    output = model(input)
    print(output.shape)
    pass