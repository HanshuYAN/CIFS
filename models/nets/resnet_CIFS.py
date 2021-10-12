'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
from torch import masked_fill
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from .resnet import BasicBlock, Bottleneck

class Global_Avg_Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1,1))
        out = out.view(out.size(0), -1)
        return out

class nlBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(nlBlock, self).__init__()
        self.global_avg_flatten = Global_Avg_Flatten()
        self.fc = nn.Linear(in_planes, planes)
        self.softplus = nn.Softplus(beta=5)
    def forward(self, x):
        out = self.softplus( self.fc(self.global_avg_flatten(x)) )
        return out


class CIFS(nn.Module):
    def __init__(self, n_feat=512, n_cls=10, mode='linear'):
        super().__init__()
        if mode == 'linear':
            self.Probe = nn.Sequential(*[Global_Avg_Flatten(), nn.Linear(n_feat, n_cls)])
        else:
            self.Probe = nn.Sequential(*[nlBlock(n_feat, 128), nn.Linear(128, n_cls)])

    def forward(self, feat, y=None):
        ''' # y: (batch), feat: (batch, 512, h, w); ## masked feat: (batch, 10), cas prediction: (batch, 512) '''
        Mask = self._get_mask_with_graph(feat, y)
        pred_raw = self.Probe(feat) # (batch, 10)
        masked_feat = feat * Mask
        return masked_feat, pred_raw
    
    def _get_mask_with_graph(self, feat, y=None):
        N, C, _, _ = feat.shape
        feat = feat.detach().clone()
        feat.requires_grad_(True)
        
        logits = self.Probe(feat) # (batch, 10)
        if not self.training:
            pred = logits.topk(k=2, dim=1)[1]
            pred_t1 = pred[:,0]; pred_t2 = pred[:,1]
            top1_logit = logits[torch.tensor(list(range(N))), pred_t1].sum()
            top2_logit = logits[torch.tensor(list(range(N))), pred_t2].sum()
        else:
            pred = logits.topk(k=2, dim=1)[1]
            pred_t2 = pred[:,1]
            top1_logit = logits[torch.tensor(list(range(N))), y].sum()
            top2_logit = logits[torch.tensor(list(range(N))), pred_t2].sum()
            
        max_logit = top1_logit + top2_logit
        mask = autograd.grad(max_logit, feat, create_graph=True)[0]
        mask = F.adaptive_avg_pool2d(mask, (1,1)) * mask.size(2) * mask.size(3)
        mask = F.softmax(mask.view(N,C), dim=1)
        # mask = F.softmax(mask.view(N,C), dim=1)*2
        return mask.view(N, C, 1, 1)
    
    def _requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad
            

class CIFSBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, mode='linear'):
        super(CIFSBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.Probe = CIFS(planes, 10, mode)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x, label=None):
        feat = F.relu(self.bn1(self.conv1(x)))
        feat = self.bn2(self.conv2(feat))
        feat += self.shortcut(x)
        feat = F.relu(feat)
        #
        masked_feat, pred_raw = self.Probe(feat, label)
        return masked_feat, pred_raw, feat
    
    
class ResNet_L4(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_L4, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer_cas_(CIFSBlock, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_layer_cas_(self, block, planes, num_blocks, stride, modes=['nonlinear', 'linear']):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        assert len(strides) == len(modes)
        
        for i, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, mode=modes[i]))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)
    
    def forward(self, x, y=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        pred_raw_list = []
        for layer in self.layer4:
            out, pred_raw, _ = layer(out, y)
            pred_raw_list.append(pred_raw)
    
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out) # (1,512) -- (1,10)
        return out, pred_raw_list # 51.23  
    

def ResNet18_L4():
    return ResNet_L4(BasicBlock, [2,2,2,2])
    
    