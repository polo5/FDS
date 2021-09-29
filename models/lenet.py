import torch
import torch.nn as nn
import torch.nn.functional as F
from models.meta_factory import ReparamModule
from models.helpers import *

class Flatten(nn.Module):
    """
    NN module version of torch.nn.functional.flatten
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.flatten(input, start_dim=1, end_dim=-1)

class LeNet(nn.Module):
    def __init__(self, n_classes, n_channels, im_size):
        super(LeNet, self).__init__()
        assert im_size in [28, 32]
        h = 16*5*5 if im_size==32 else 16*4*4
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.im_size = im_size

        self.layers = nn.Sequential(
            nn.Conv2d(n_channels, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Flatten(),
            nn.Linear(h, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, n_classes))


    def forward(self, x):
        return self.layers(x)

class MetaLeNet(ReparamModule):
    def __init__(self, n_classes, n_channels, im_size, device='cpu'):
        super(MetaLeNet, self).__init__()
        assert im_size in [28, 32]
        h = 16*5*5 if im_size==32 else 16*4*4
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.im_size = im_size
        self.device = device # must be defined for parent class

        self.layers = nn.Sequential(
            nn.Conv2d(n_channels, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            Flatten(),
            nn.Linear(h, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, n_classes))


    def forward(self, x):
        return self.layers(x)





if __name__ == '__main__':
    import time
    from torchsummary import summary
    from utils.helpers import *

    set_torch_seeds(0)
    x = torch.FloatTensor(256, 3, 32, 32).uniform_(0, 1)


    ## Test meta LeNet
    model = MetaLeNet(n_classes=10, n_channels=3, im_size=32, device='cpu')
    weights = model.get_param()
    t0 = time.time()
    out = model.forward_with_param(x, weights)
    print(f'time for meta fw pass: {time.time() - t0}s')
    summary(model, (3, 32, 32))
















