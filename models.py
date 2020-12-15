import torch
import torch.nn as nn
import pretrainedmodels

class Seresnet_Wind(nn.Module):
    def __init__(self, out_dim = 1, pretrained = True):
        super(Seresnet_Wind, self).__init__()
        if pretrained:
            self.model_body = nn.Sequential(
                *list(pretrainedmodels.__dict__["se_resnext50_32x4d"](num_classes=1000, pretrained="imagenet").children())[
                    :-2
                ]
            )
        else:
            self.model_body = nn.Sequential(
            *list(pretrainedmodels.__dict__["se_resnext50_32x4d"](num_classes=1000, pretrained=None).children())[
                :-2
            ]
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fea_bn = nn.BatchNorm1d(2048)
        self.fea_bn.bias.requires_grad_(False)
        self.head = nn.Linear(2048, out_dim)

    def forward(self, x):
        x = self.model_body(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fea_bn(x)
        out = self.head(x)
        return out