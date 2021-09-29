"""
Base architecture taken from https://github.com/xternalz/WideResNet-pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.meta_factory import ReparamModule


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate):
        super(BasicBlock, self).__init__()
        self.dropRate = dropRate
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if self.equalInOut:
            out = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(out)))
        else: #keep x var so can add it in skip connection
            x = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(x)))

        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)
        out = self.conv2(out)

        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, n_classes, n_channels, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(n_channels, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)

        # global average pooling and classifier
        self.final_bn = nn.BatchNorm2d(nChannels[3], affine=True)
        self.final_relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], n_classes)
        self.nChannels = nChannels[3]


    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.final_relu(self.final_bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

class MetaWideResNet(ReparamModule):
    def __init__(self, depth, n_classes, n_channels, widen_factor=1, dropRate=0.0, device='cpu'):
        super(MetaWideResNet, self).__init__()
        self.device = device
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(n_channels, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)

        # global average pooling and classifier
        self.final_bn = nn.BatchNorm2d(nChannels[3], affine=True)
        self.final_relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], n_classes)
        self.nChannels = nChannels[3]

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.final_relu(self.final_bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


if __name__ == '__main__':
    import time
    from torchsummary import summary
    from utils.helpers import *

    set_torch_seeds(0)
    x = torch.FloatTensor(2, 3, 32, 32).uniform_(0, 1)

    ## Test normal WRN
    model = WideResNet(depth=40, widen_factor=2, n_channels=3, n_classes=10, dropRate=0.0)
    t0 = time.time()
    out = model(x)
    print(f'time for normal fw pass: {time.time() - t0}s')
    summary(model, (3, 32, 32))

    ## Test meta WRN
    model = MetaWideResNet(depth=40, widen_factor=2, n_channels=3, n_classes=10, device='cpu')
    weights = model.get_param()
    t0 = time.time()
    out = model.forward_with_param(x, weights)
    print(f'time for meta fw pass: {time.time() - t0}s')
    summary(model, (3, 32, 32))


