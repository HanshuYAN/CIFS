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


class CSAFR(nn.Module):
    def __init__(self, n_feat=512, n_cls=10, mode='linear'):
        super().__init__()
        if mode == 'linear':
            self.Probe = nn.Sequential(*[Global_Avg_Flatten(), nn.Linear(n_feat, n_cls)])
        else:
            self.Probe = nn.Sequential(*[nlBlock(n_feat, 128), nn.Linear(128, n_cls)])

    def forward(self, feat, y=None):
        ''' # y: (batch), feat: (batch, 512, h, w); ## masked feat: (batch, 10), cas prediction: (batch, 512) '''
        Mask = self._get_mask_with_graph(feat, y) # Here Mask are detach, no gradient
        pred_cas = self.Probe(feat) # (batch, 10)
        masked_feat = feat * Mask
        return masked_feat, pred_cas
    
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
            
            
        mask1 = autograd.grad(top1_logit, feat, create_graph=True)[0]
        mask1 = F.adaptive_avg_pool2d(mask1, (1,1)) * mask1.size(2) * mask1.size(3)
        mask1 = F.softmax(mask1.view(N,C), dim=1)
        
        mask2 = autograd.grad(top2_logit, feat, create_graph=True)[0]
        mask2 = F.adaptive_avg_pool2d(mask2, (1,1)) * mask2.size(2) * mask2.size(3)
        mask2 = F.softmax(mask2.view(N,C), dim=1)
        
        mask = mask1 + mask2
        
        return mask.view(N, C, 1, 1)
    
    def _requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad
            

## CAS module official

class CSAFRBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, mode='linear'):
        super(CSAFRBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.Probe = CSAFR(planes, 10, mode)

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
        masked_feat, pred_cas = self.Probe(feat, label)
        return masked_feat, pred_cas, feat
    
    
class ResNet_L4(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_L4, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer_cas_(CSAFRBlock, 512, num_blocks[3], stride=2)
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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        N, C, _, _ = x.shape
        feat_l3 = F.adaptive_avg_pool2d(out.detach().clone(), 1).view(N, -1)
        
        # --- CAS ---
        for layer in self.layer4:
            out, pred_cas, feat_l4_unmasked = layer(out, y)
        # --- CAS ---
        
        out = F.avg_pool2d(out, 4)
        feat_l4_masked = out.view(out.size(0), -1)
        out = self.linear(feat_l4_masked) # (1,512) -- (1,10)
        
        feat_l4_unmasked = F.avg_pool2d(feat_l4_unmasked, 4)
        feat_l4_unmasked = feat_l4_unmasked.view(feat_l4_unmasked.size(0), -1)
        
        if which_feat == 'l3':
            return out, feat_l3
        elif which_feat == 'l4_unmasked':
            return out, feat_l4_unmasked
        elif which_feat == 'l4_masked':
            return out, feat_l4_masked
        else:
            assert False
            
    

def ResNet18_L4():
    return ResNet_L4(BasicBlock, [2,2,2,2])
    
    
    
    


class ResNet10_3_4(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_channels=3):
        super(ResNet10_3_4, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        
        self.Probe_nL = CSAFR(256, num_classes, mode='nolinear')
        self.Probe_L = CSAFR(512, num_classes, mode='linear')
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    
    def forward(self, x, y=None):
        pred_cas_list = []
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out, pred_cas = self.Probe_nL(out, y); pred_cas_list.append(pred_cas)
        
        out = self.layer4(out)
        
        out, pred_cas = self.Probe_L(out, y); pred_cas_list.append(pred_cas)
    
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out) # (1,512) -- (1,10)
        return out, pred_cas_list
    
    
    def predict_with_feats(self, x, y=None, which_feat=None):
        unmasked_feats = []
        masked_feats = []
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        
        out = self.layer3(out); unmasked_feats.append(out.detach().clone())
        out, pred_cas = self.Probe_nL(out, y); masked_feats.append(out.detach().clone())
        
        out = self.layer4(out); unmasked_feats.append(out.detach().clone())
        out, pred_cas = self.Probe_L(out, y); masked_feats.append(out.detach().clone())
        
    
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out) # (1,512) -- (1,10)
        
        if which_feat == 'probe_0':
            return out, unmasked_feats[-1]
        elif which_feat == 'probe_0_masked':
            return out, masked_feats[-1]
        elif which_feat == 'probe_1':
            return out, unmasked_feats[-2]
        elif which_feat == 'probe_1_masked':
            return out, masked_feats[-2]
        else:
            assert False
            
    

def ResNet_Gray_10(num_classes=10):
    return ResNet10_3_4(BasicBlock, [1,1,1,1], num_classes=num_classes, in_channels=1)


    
if __name__ == '__main__':
    def test():
        # net = ResNet18()
        # y = net(Variable(torch.randn(1,3,64,64)))
        net = CSAFR(512, 10)
        # net.eval()
        feat = autograd.Variable(torch.randn(5,512,16,16))
        label = autograd.Variable(torch.randint(10, (5,)))

        # net.fc.weight.data.fill_(1)
        out = net(feat, label)
        # out = net._get_mask_undetached(feat, label)
        # print(y.size())
        # print(net)
        
        # f = Variable(torch.randn(5,512,4,4))
        # cas = Variable(torch.randn(5,512))
        pass
    test()
    
    