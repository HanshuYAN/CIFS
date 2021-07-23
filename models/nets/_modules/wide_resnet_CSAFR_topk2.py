import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from .wide_resnet import BasicBlock, NetworkBlock

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
        elif mode=='nonlinear':
            self.Probe = nn.Sequential(*[nlBlock(n_feat, 240), nn.Linear(240, n_cls)])
        else:
            assert False

    def forward(self, feat, y=None):
        ''' # y: (batch), feat: (batch, 512, h, w); ## masked feat: (batch, 10), cas prediction: (batch, 512) '''
        Mask = self._get_mask_with_graph(feat, y) # Here Mask are detach, no gradient
        pred_Probe = self.Probe(feat) # (batch, 10)
        masked_feat = feat * Mask
        return masked_feat, pred_Probe
    
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
        mask = F.softmax(mask.view(N,C), dim=1) * 2
        # mask = F.softmax(mask.view(N,C), dim=1)
        return mask.view(N, C, 1, 1)
    
    
    
class BasicBlock_CSAFR(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, num_classes=10):
        super(BasicBlock_CSAFR, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        # Probe
        self.Probe = CSAFR(in_planes, num_classes, mode='nonlinear')

    def forward(self, x, label=None):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))

        input = out if self.equalInOut else x
        
        # --- CAS ---
        input, pred_Probe = self.Probe(input, label)
        # --- CAS ---
        
        out = self.relu2(self.bn2(self.conv1(input)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        out = torch.add(x if self.equalInOut else self.convShortcut(x), out)
        return out, pred_Probe


class NetworkBlock_CSAFR(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, num_classes=10):
        super(NetworkBlock_CSAFR, self).__init__()
        self.nb_layers = nb_layers
        self.layer = self._make_layer(BasicBlock_CSAFR, in_planes, out_planes, nb_layers, stride, dropRate, num_classes)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, num_classes):
        layers = []
        for i in range(int(nb_layers)-1):
            layers.append(BasicBlock(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
            
        for i in range(int(nb_layers)-1, int(nb_layers)):
            layers.append(BasicBlock_CSAFR(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, num_classes))
        return nn.ModuleList(layers)

    def forward(self, x, y=None):
        Probe_out_list = []
        out = x
        for i in range(int(self.nb_layers)-1):
            out = self.layer[i](out)
            
        for i in range(int(self.nb_layers)-1, int(self.nb_layers)):
             out, Probe_out = self.layer[i](out, y)
             Probe_out_list.append(Probe_out)

        return out, Probe_out_list



class Wide_ResNet(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0):
        super(Wide_ResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # # 1st sub-block
        # self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock_CSAFR(n, nChannels[2], nChannels[3], block, 2, dropRate, num_classes)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        
        # Probe
        self.Probe = CSAFR(nChannels[3], num_classes, mode='linear')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, y=None):
        Probe_out_list = []
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        # out = self.block3(out)
        out, Probe_out_list = self.block3(out, y)
        out = self.relu(self.bn1(out))


        out, pred_Probe = self.Probe(out, y)
        Probe_out_list.append(pred_Probe)
        
        # fc_in = torch.mean(out.view(out.shape[0], out.shape[1], -1), dim=-1)
        # fc_out = self.extra_fc(fc_in.view(out.shape[0], out.shape[1]))
        # Probe_out_list.append(fc_out)
        # if self.training:
        #     N, C, H, W = out.shape
        #     mask = self.extra_fc.weight[y, :]
        #     out = out * mask.view(N, C, 1, 1)
        # else:
        #     N, C, H, W = out.shape
        #     pred_label = torch.max(fc_out, dim=1)[1]
        #     mask = self.extra_fc.weight[pred_label, :]
        #     out = out * mask.view(N, C, 1, 1)

        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out), Probe_out_list
