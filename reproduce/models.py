import torch
import torch.nn as nn

from torchvision import models

from utils import Swish_Module

'''
Assume all with pretrained now
'''

class ResNetExample(nn.Module):
    def __init__(self):
        super(ResNetExample, self).__init__()
        self.model = models.resnet152(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(100,1)
        )
    def forward(self, x):
        return self.model(x)