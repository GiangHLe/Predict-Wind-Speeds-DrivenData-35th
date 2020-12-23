import torch
import torch.nn as nn

from torchvision import models
import pretrainedmodels

from utils import Swish_Module

'''
Assume all with pretrained now which leads to all use Batch Norm and 3 channels image
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

class Seresnext_Wind(nn.Module):
    def __init__(self, pretrained = True):
        super(Seresnext_Wind, self).__init__()
        if pretrained:
            self.extract = nn.Sequential(
                *list(pretrainedmodels.__dict__["se_resnext101_32x4d"](num_classes=1000, pretrained="imagenet").children())[
                    :-2
                ]
            )
        else:
            self.extract = nn.Sequential(
            *list(pretrainedmodels.__dict__[name](num_classes=1000, pretrained=None).children())[
                :-2
            ]
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Dropout(p = 0.2),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.Dropout(p = 0.2),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.Dropout(p = 0.2),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.extract(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        out = self.head(x)
        return out