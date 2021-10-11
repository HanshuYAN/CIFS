import os, argparse, pathlib, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models.BaseModel import BaseModelDNN
from metric.classification import benchmark_defense_success_rate, dataset_accuracy
from utils import timer, get_epoch_logger

parser = argparse.ArgumentParser(description='Configuration')
parser.add_argument('--SEED', default=0, type=int)
parser.add_argument('--GPU_IDs', nargs='+', default=[0], type=int)
parser.add_argument('--is_Train', action='store_true')
parser.add_argument('--network', default='Vanilla', type=str)
args, _ = parser.parse_known_args()
if args.is_Train:
    parser.add_argument('--exp_path', default='./experiments/...')
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=250, type=int)
    
    parser.add_argument('--is_AdvTr', action='store_true')
    parser.add_argument('--attack_loss', default='Joint', choices=['CE', 'Joint'])
    parser.add_argument('--beta_cls', default=2, type=float)
    parser.add_argument('--beta_atk', default=2, type=float)
    parser.add_argument('--tr_epochs', default=120, type=int)

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
    parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float, metavar='W')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
    parser.add_argument('--milestones', nargs='+', default=[75, 90], type=int)
    
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint', type=str, default='./experiments/')
    parser.add_argument('--net_only', default=True, type=lambda x: bool(int(x)))
else:
    pass
    
args = parser.parse_args()
    
np.random.seed(args.SEED)
torch.manual_seed(args.SEED)

if args.network == '_':
    from models.nets.resnet_ import ResNet18
    ResNet18 = ResNet18
elif args.network == 'CAS_L4':
    from models.nets.resnet_CAS import ResNet18_L4
    ResNet18 = ResNet18_L4
elif args.network == 'CIFS_L4':
    from models.nets.resnet_CIFS import ResNet18_L4
    ResNet18 = ResNet18_L4
else:
    assert False


class joint_CE_loss(nn.Module):
    def __init__(self, beta=2, is_joint=True) -> None:
        super().__init__()
        self.beta = beta
        self.is_joint = is_joint
        
    def forward(self, logits_tuple, target):
        logits_final, logits_raw_list = logits_tuple
        loss = 0
        if self.is_joint:
            if len(logits_raw_list) > 0:
                for logits in logits_raw_list:
                    loss += F.cross_entropy(logits, target)
                loss = (self.beta/len(logits_raw_list)) * loss
        loss += F.cross_entropy(logits_final, target)
        
        loss = loss * 3/(self.beta + 1)
        return loss


class CIFAR10_Classifier(BaseModelDNN):
    def __init__(self, args=None, device='cuda', is_train=False) -> None:
        super().__init__()

        self.net = ResNet18().to(device)

        self.device = device
        self.GPU_IDs = args.GPU_IDs
        if len(self.GPU_IDs) > 1:
            self.net = nn.DataParallel(module=self.net, device_ids=self.GPU_IDs)
        
        if is_train:
            self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            self.criterion = joint_CE_loss(beta=args.beta_cls)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.milestones, gamma=0.1)
            self.tr_epochs = args.tr_epochs
            self.start_epoch = 0
            self.log_interval = 100
        else:
            self.eval_mode()
            self.set_requires_grad([self.net], False)
    
    def eval_mode(self):
        self.net.eval()
    def train_mode(self):
        self.net.train()
        
    def load_networks(self, path):
        self.checkpoint = torch.load(path)
        if len(self.GPU_IDs) == 1:
            self.net.load_state_dict(self.checkpoint['state_dict'])
        else:
            self.net.module.load_state_dict(self.checkpoint['state_dict'])
        
    def resume_training(self, path, net_only=True):
        self.load_networks(path)
        if not net_only:
            self.start_epoch = self.checkpoint['stop_epoch']
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            self.scheduler.last_epoch = self.start_epoch
    
    def predict(self, x):
        assert not self.net.training
        logits_ce, logits_cas = self.net(x)
        return logits_ce
    
           
    def _fit_one_epoch(self, train_loader, epoch, logger, is_AdvTr=False, adversary=None, device=torch.device("cuda")):
        tic_toc = timer()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            if is_AdvTr:
                self.eval_mode()
                self.set_requires_grad([self.net], False)
                data = adversary.perturb(data, target)
                self.set_requires_grad([self.net], True)
                self.train_mode()
                
            self.optimizer.zero_grad()
            logits_ce, logits_CAS = self.net(data, target)
            loss = self.criterion((logits_ce, logits_CAS), target)
            loss.backward()
            self.optimizer.step()
            
            if (batch_idx+1) % self.log_interval == 0:
                logger.info('[{}/{} ({:.0f}%)], Loss: {:.6f}, Time for Batches: {:03f}'.format(
                    batch_idx *len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), 
                    loss.item(), 
                    tic_toc.toc()))
                
        if self.scheduler is not None:
            self.scheduler.step()
            
             
    def _evaluate(self, is_AdvTr, epoch, test_loader, attack_test, attack_kwargs_test):
        self.eval_mode()
        if is_AdvTr:
            if epoch == 0 or (epoch+1) % 10 == 0 or (epoch >= 50 and (epoch+1)%2==0):
                accuracy, defense_success_rate, message, _ = benchmark_defense_success_rate(self.predict,
                    self.net, test_loader, attack_test, attack_kwargs_test, device="cuda")
            else:
                accuracy = 0; defense_success_rate = 0; message = '- evaluation skipped!'

            return defense_success_rate, message
        else:
            accuracy, message = dataset_accuracy(self.predict, test_loader=test_loader)
            return accuracy, message
                
            
    def fit(self, train_loader, test_loader, logger, save_path='.', is_AdvTr=False, attack_tr=None, attack_kwargs_tr=None, attack_test=None, attack_kwargs_test=None):        
        tic_toc = timer()
        epoch_logger = get_epoch_logger()
        adversary = attack_tr(self.net, **attack_kwargs_tr) if is_AdvTr else None
        
        for epoch in range(self.start_epoch, self.tr_epochs):
            logger.info('Training Epoch: {}; Learning rate: {:0.8f}  .....'.format(epoch, self.optimizer.param_groups[0]['lr']))
            self.train_mode()
            self._fit_one_epoch(train_loader, epoch, logger=logger, is_AdvTr=is_AdvTr, adversary=adversary, device=self.device)
                             
            result, message = self._evaluate(is_AdvTr, epoch, test_loader, attack_test, attack_kwargs_test)
            logger.info(message + '\t Time for an epoch: {:.2f}'.format(tic_toc.toc()))
            
            epoch_logger.append_results([epoch, result])
            best_epoch = epoch_logger.update_best_epoch_to_logger(logger)
            
            model_state_dict = self.net.state_dict() if len(self.GPU_IDs)<2 else self.net.module.state_dict()
            checkpoint = {'state_dict':model_state_dict, 'stop_epoch':epoch, 'optimizer': self.optimizer.state_dict()}
            
            torch.save(checkpoint,os.path.join(save_path, 'ckp_latest.pt'))
            
            if best_epoch == epoch:
                torch.save(checkpoint, os.path.join(save_path, "ckp_best.pt"))
            
            if is_AdvTr and epoch>=49 and (epoch+1)%2==0:
                torch.save(checkpoint,os.path.join(save_path, 'ckp_{:03d}.pt'.format(epoch)))


            
if __name__ == '__main__':
    if not args.is_Train:
        pass
    else:
        from datasets import get_cifar10_train_loader, get_cifar10_test_loader
        from utils import get_logger
        pathlib.Path(os.path.join(args.exp_path, 'nets')).mkdir(parents=True, exist_ok=True)
        logger = get_logger(os.path.join(args.exp_path, 'logging.txt'))
        logger.info(args)
            
        # Tasks: Dataset and Model
        train_loader = get_cifar10_train_loader(batch_size=args.train_batch_size, shuffle=True)
        test_loader = get_cifar10_test_loader(batch_size=args.test_batch_size, shuffle=False)
        model = CIFAR10_Classifier(args, is_train=True)
        logger.info(model.net)
        if args.resume:
            model.resume_training(path=args.checkpoint, net_only=args.net_only)
        
        # Define Adversary
        from advertorch.attacks import LinfPGDAttack
        if not args.is_AdvTr:
            attack_tr = None; attack_kwargs_tr = None
            attack_test = None; attack_kwargs_test = None
        else:
            attack_tr = LinfPGDAttack
            attack_kwargs_tr = dict(loss_fn=joint_CE_loss(beta=args.beta_cls, is_joint=True if args.attack_loss == 'Joint' else False), 
                                             eps=8/255, nb_iter=10, eps_iter=2/255, 
                                             rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
            attack_test = LinfPGDAttack
            attack_kwargs_test = dict(loss_fn=joint_CE_loss(beta=args.beta_atk, is_joint=True if args.attack_loss == 'Joint' else False), 
                                    eps=8/255, nb_iter=20, eps_iter=0.1 * (8/255), 
                                    rand_init=False, clip_min=0.0, clip_max=1.0, targeted=False)

        # Training
        model.fit(train_loader=train_loader, test_loader=test_loader,
                  logger=logger, save_path=os.path.join(args.exp_path, 'nets'),
                  is_AdvTr=args.is_AdvTr, 
                  attack_tr=attack_tr, attack_kwargs_tr=attack_kwargs_tr, 
                  attack_test=attack_test, attack_kwargs_test=attack_kwargs_test)
        

