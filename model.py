import torch
import torch.nn as nn
from torch.utils import data
import torchvision
import torchvision.transforms as transforms

class Resnet50(nn.Module):
    '''
    Resnet architecture for multi-label ingredient classification
    '''
    def __init__(self, n_classes, is_pretrained=True):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=is_pretrained)
        resnet.fc = nn.Sequential(
            nn.BatchNorm1d(resnet.fc.in_features),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(resnet.fc.in_features, n_classes),
        )
        self.base_model = resnet
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return self.sigm(self.base_model(x))

class Resnet50_baseline(nn.Module):
    '''
    Resnet architecture for multi-label ingredient classification
    '''
    def __init__(self, is_pretrained=True):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=is_pretrained)
        self.base_model = resnet
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        ret = self.sigm(self.base_model(x))
        print(ret.shape)
        return ret
