import torch
import torch.nn as nn
import pretrainedmodels
from utils import init_weights

class Seresnet_Wind(nn.Module):
    def __init__(self, type = 1, out_dim = 1, pretrained = True, gray = False):
        super(Seresnet_Wind, self).__init__()
        if type == 1:
            name = "se_resnext50_32x4d"
        else:
            name = "se_resnext101_32x4d"
        if pretrained:
            self.model_body = nn.Sequential(
                *list(pretrainedmodels.__dict__[name](num_classes=1000, pretrained="imagenet").children())[
                    :-2
                ]
            )
        else:
            self.model_body = nn.Sequential(
            *list(pretrainedmodels.__dict__[name](num_classes=1000, pretrained=None).children())[
                :-2
            ]
        )
        if gray:
            # print(self.model_body)
            self.model_body[0].conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        # self.fea_bn = nn.BatchNorm1d(2048)
        # self.fea_bn.bias.requires_grad_(False)
        self.head = nn.Linear(2048, out_dim)
        if not pretrained:
            self.model_body.apply(init_weights)
            self.head.apply(init_weights)

    def forward(self, x):
        x = self.model_body(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        # x = self.fea_bn(x)
        out = self.head(x)
        return out