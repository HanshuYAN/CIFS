
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

class joint_CAS_loss(nn.Module):
    def __init__(self, beta=2, is_joint=True) -> None:
        super().__init__()
        self.beta = beta
        self.is_joint = is_joint
        
    def forward(self, logits_tuple, target):
        logits_ce, logits_cas_list = logits_tuple
        loss = 0
        if self.is_joint:
            if len(logits_cas_list) > 0:
                for logits in logits_cas_list:
                    loss += F.cross_entropy(logits, target)
                loss = (self.beta/len(logits_cas_list)) * loss
        loss += F.cross_entropy(logits_ce, target)
        return loss
    
    
class CW_loss(nn.Module):
    def __init__(self, beta=2, is_joint=True):
        super().__init__()
        self.is_joint = is_joint
        self.beta = beta
        
    def forward(self, logits_tuple, target):
        logits_ce, logits_cas_list = logits_tuple
        loss = 0
        if self.is_joint:
            if len(logits_cas_list) > 0:
                for logits in logits_cas_list:
                    loss += self._cw_loss(logits, target)
                loss = (self.beta/len(logits_cas_list)) * loss
        loss += self._cw_loss(logits_ce, target)
        return loss

    def _cw_loss(self, output, target,confidence=50, num_classes=10):
        # Compute the probability of the label class versus the maximum other
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
    
    
    
#
#   
# NEW version losses
#
#
class joint_CE_loss(nn.Module):
    def __init__(self, is_attack, is_joint, logit_index=0, beta=2) -> None:
        super().__init__()
        self.is_atk = is_attack
        self.is_joint = is_joint
        self.beta = beta
        self.logit_index = logit_index
        
    def forward(self, logits_tuple,  target):
        logits_ce, logits_cas_list = logits_tuple
        
        if self.is_atk == True:
            if self.is_joint == False:
                logits = [logits_ce] + logits_cas_list
                loss = F.cross_entropy(logits[self.logit_index], target)
            else:
                loss = self.get_joint_loss(logits_tuple, target)
        else:
            loss = self.get_joint_loss(logits_tuple, target)            
        return loss
    
    def get_joint_loss(self, logits_tuple, target):
        logits_ce, logits_cas_list = logits_tuple
        loss = 0
        if len(logits_cas_list) > 0:
            for logits in logits_cas_list:
                loss += F.cross_entropy(logits, target)
            loss = (self.beta/len(logits_cas_list)) * loss
        loss += F.cross_entropy(logits_ce, target)
        return loss
        

class joint_CW_loss(nn.Module):
    def __init__(self, is_attack, is_joint, logit_index=0, beta=2) -> None:
        super().__init__()
        self.is_atk = is_attack
        self.is_joint = is_joint
        self.beta = beta
        self.logit_index = logit_index
        
    def forward(self, logits_tuple,  target):
        logits_ce, logits_cas_list = logits_tuple
        
        if self.is_atk == True:
            if self.is_joint == False:
                logits = [logits_ce] + logits_cas_list
                loss = F.cross_entropy(logits[self.logit_index], target)
            else:
                loss = self.get_joint_loss(logits_tuple, target)
        else:
            loss = self.get_joint_loss(logits_tuple, target)            
        return loss
    
    def get_joint_loss(self, logits_tuple, target):
        logits_ce, logits_cas_list = logits_tuple
        loss = 0
        if len(logits_cas_list) > 0:
            for logits in logits_cas_list:
                loss += self._cw_loss(logits, target)
            loss = (self.beta/len(logits_cas_list)) * loss
        loss += self._cw_loss(logits_ce, target)
        return loss
    
    def _cw_loss(self, output, target,confidence=50, num_classes=10):
        # Compute the probability of the label class versus the maximum other
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