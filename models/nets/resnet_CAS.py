'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .resnet import BasicBlock, Bottleneck

class CAS(nn.Module):
    def __init__(self, n_feat=512, n_cls=10):
        super().__init__()
        self.n_cls = n_cls
        self.fc = nn.Linear(n_feat, n_cls) 

    def forward(self, feat, y=None):
        """ # y: (batch), feat: (batch, 512, h, w); ## masked feat: (batch, 10), cas prediction: (batch, 512) """
        pred_cas = F.adaptive_avg_pool2d(feat, (1,1))
        pred_cas = pred_cas.view(pred_cas.size(0), -1) # (batch, 512)
        pred_cas = self.fc(pred_cas) # (batch, 10)
        if self.training:
            Mask = self.fc.weight[y, :]
            N, C, _, _ = feat.shape
            masked_feat = feat * Mask.view(N, C, 1, 1)
            return masked_feat, pred_cas
        else:
            y_pred = pred_cas.max(1, keepdim=False)[1]
            Mask = self.fc.weight[y_pred, :]
            N, C, _, _ = feat.shape
            masked_feat = feat * Mask.view(N, C, 1, 1)
            return masked_feat, pred_cas

class CASBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, num_classes=10):
        super(CASBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.CAS = CAS(planes, num_classes)

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
        # CAS
        masked_feat, pred_cas = self.CAS(feat, label)
        return masked_feat, pred_cas, feat
    
    
class ResNet_L4(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_channels=3):
        super(ResNet_L4, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer_cas_(CASBlock, 512, num_blocks[3], stride=2, num_classes=num_classes)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_layer_cas_(self, block, planes, num_blocks, stride, num_classes):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, num_classes))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)
    
    def forward(self, x, y=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # --- CAS ---
        pred_cas_list = []
        for layer in self.layer4:
            out, pred_cas, _ = layer(out, y)
            pred_cas_list.append(pred_cas)
        # --- CAS ---
    
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out) # (1,512) -- (1,10)
        return out, pred_cas_list     
            
    def predict_with_feats(self, x, y=None, which_feat=None):
        unmasked_feats = []
        masked_feats = []
        masks = []
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        for layer in self.layer4:
            out, pred_cas, out_unmasked = layer(out, y)
            
            unmasked_feats.append(out_unmasked.detach().clone())
            masked_feats.append(out.detach().clone())
            # masks.append(mask.detach().clone())
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out) # (1,512) -- (1,10)
            
        if which_feat == 'probe_0':
            return out, unmasked_feats[0]
        elif which_feat == 'probe_0_masked':
            return out, masked_feats[0]
        # elif which_feat == 'probe_0_mask':
        #     return out, masks[-1]
        
        elif which_feat == 'probe_1':
            return out, unmasked_feats[1]
        elif which_feat == 'probe_1_masked':
            return out, masked_feats[1]
        # elif which_feat == 'probe_1_mask':
        #     return out, masks[-2]
        else:
            assert False
    
def ResNet18_L4(num_classes=10):
    return ResNet_L4(BasicBlock, [2,2,2,2], num_classes=num_classes)




class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        
        self.CAS = CAS(512,num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, y=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # --- CAS ---
        if self.training:
            out, pred_cas = self.CAS(feat=out, y=y) # masked feat, pred of cas
        else:
            out, pred_cas = self.CAS(feat=out)
        # --- CAS ---
    
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out) # (1,512) -- (1,10)
        return out, [pred_cas]
    
def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])