import os, argparse, pathlib, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from models.BaseModel import BaseModelDNN
from utils import timer, get_epoch_logger

parser = argparse.ArgumentParser(description='Configuration')
parser.add_argument('--SEED', default=0, type=int)
parser.add_argument('--GPU_IDs', nargs='+', default=[0], type=int)
parser.add_argument('--is_Train', action='store_true')
parser.add_argument('--network', default='_', type=str)
args, _ = parser.parse_known_args()
if args.is_Train:
    pass
else:
    parser.add_argument('--which_model', default='./experiments/cifar10_resnet18/nets/ckp_best.pt')
    parser.add_argument('--is_attack', default=True, type=lambda x: bool(int(x)))
    parser.add_argument('--is_joint', default=None, type=lambda x: bool(int(x)))
    parser.add_argument('--logit_index', default=None, type=int)
    parser.add_argument('--beta_atk', default=None, type=float)
    parser.add_argument('--filename', default='results.txt')
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
    def __init__(self, is_joint, logit_index=0, beta=2) -> None:
        super().__init__()
        self.is_joint = is_joint
        self.beta = beta
        self.logit_index = logit_index
        
    def forward(self, logits_tuple,  target):
        logits_final, logits_raw_list = logits_tuple
        if self.is_joint == False:
            logits = [logits_final] + logits_raw_list
            loss = F.cross_entropy(logits[self.logit_index], target)
        else:
            loss = self.get_joint_loss(logits_tuple, target)           
        return loss
    
    def get_joint_loss(self, logits_tuple, target):
        logits_final, logits_raw_list = logits_tuple
        loss = 0
        if len(logits_raw_list) > 0:
            for logits in logits_raw_list:
                loss += F.cross_entropy(logits, target)
            loss = (self.beta/len(logits_raw_list)) * loss
        loss += F.cross_entropy(logits_final, target)
        return loss
        

class joint_CW_loss(nn.Module):
    def __init__(self, is_attack, is_joint, logit_index=0, beta=2) -> None:
        super().__init__()
        self.is_joint = is_joint
        self.beta = beta
        self.logit_index = logit_index
        
    def forward(self, logits_tuple,  target):
        logits_final, logits_raw_list = logits_tuple
        if self.is_joint == False:
            logits = [logits_final] + logits_raw_list
            loss = F.cross_entropy(logits[self.logit_index], target)
        else:
            loss = self.get_joint_loss(logits_tuple, target)
        return loss
    
    def get_joint_loss(self, logits_tuple, target):
        logits_final, logits_raw_list = logits_tuple
        loss = 0
        if len(logits_raw_list) > 0:
            for logits in logits_raw_list:
                loss += self._cw_loss(logits, target)
            loss = (self.beta/len(logits_raw_list)) * loss
        loss += self._cw_loss(logits_final, target)
        return loss
    
    def _cw_loss(self, output, target,confidence=50, num_classes=10):
        # The same implementation as in repo CAT https://github.com/sunblaze-ucb/curriculum-adversarial-training-CAT
        target = target.data
        target_onehot = torch.zeros(target.size() + (num_classes,))
        target_onehot = target_onehot.cuda()
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        target_var = Variable(target_onehot, requires_grad=False)
        real = (target_var * output).sum(1)
        other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
        loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
        loss = torch.sum(loss)
        return loss
    

class CIFAR10_Classifier(BaseModelDNN):
    def __init__(self, args=None, device='cuda', is_train=False) -> None:
        super().__init__()
        self.net = ResNet18().to(device)
        self.device = device
        self.GPU_IDs = args.GPU_IDs
        if len(self.GPU_IDs) > 1:
            self.net = nn.DataParallel(module=self.net, device_ids=self.GPU_IDs)
        self.eval_mode()
        self.set_requires_grad([self.net], False)
    
    def eval_mode(self):
        self.net.eval()
        
    def load_networks(self, path):
        self.checkpoint = torch.load(path)
        if len(self.GPU_IDs) == 1:
            self.net.load_state_dict(self.checkpoint['state_dict'])
        else:
            self.net.module.load_state_dict(self.checkpoint['state_dict'])
    
    def predict(self, x):
        assert not self.net.training
        logits_final, logits_raw = self.net(x)
        return logits_final
        # return logits_raw[0]
    
    def predict_atk(self, x):
        assert not self.net.training
        logits_final, logits_raw = self.net(x)
        return logits_final, logits_raw 

            
if __name__ == '__main__':
    if not args.is_Train:
        from datasets import get_cifar10_test_loader
        from utils import get_logger
        test_loader = get_cifar10_test_loader(batch_size=500, sample_class=None)
        model = CIFAR10_Classifier(args); model.load_networks(args.which_model)
        
        logger = get_logger(os.path.join('./results', args.filename))
        logger.info(args.which_model + '\t***\t' + str({'is_joint':args.is_joint, 'logit_index':args.logit_index, 'beta_atk':args.beta_atk}))
        
        if True:
            from advertorch.attacks import FGSM, LinfPGDAttack
            lst_attack = [
                (FGSM, dict(
                    loss_fn=joint_CE_loss(is_joint=args.is_joint, logit_index=args.logit_index, beta=args.beta_atk),
                    eps=8/255,
                    clip_min=0.0, clip_max=1.0, targeted=False)),
                # (LinfPGDAttack, dict(
                #     loss_fn=joint_CE_loss(is_joint=args.is_joint, logit_index=args.logit_index, beta=args.beta_atk), 
                #     eps=8/255, nb_iter=20, eps_iter=0.1*(8/255), rand_init=False,
                #     clip_min=0.0, clip_max=1.0, targeted=False)),
                # (LinfPGDAttack, dict(
                #     loss_fn=joint_CE_loss(is_joint=args.is_joint, logit_index=args.logit_index, beta=args.beta_atk), 
                #     eps=8/255, nb_iter=100, eps_iter=0.1*(8/255), rand_init=False,
                #     clip_min=0.0, clip_max=1.0, targeted=False)),
                # (LinfPGDAttack, dict(
                #     loss_fn=joint_CW_loss(is_attack=args.is_attack, is_joint=args.is_joint, logit_index=args.logit_index, beta=args.beta_atk), 
                #     eps=8/255, nb_iter=30, eps_iter=0.1*(8/255), rand_init=False,
                #     clip_min=0.0, clip_max=1.0, targeted=False)),
            ]
            for attack_class, attack_kwargs in lst_attack:
                from metric.classification import topk_defense_success_rate
                message = topk_defense_success_rate(model.predict, model.predict_atk, test_loader, attack_class, attack_kwargs, device="cuda", topk=1)[-1]
                logger.info(message)
        else: 
            from metric.classification import topk_dataset_accuracy
            _, message = topk_dataset_accuracy(model.predict, test_loader, topk=1)
            logger.info(message)
        

