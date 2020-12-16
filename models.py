import torch
import torch.nn as nn
import pretrainedmodels
from utils import init_weights
from torchvision.models import resnet50

sigmoid = nn.Sigmoid()


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)


class Seresnet_Wind(nn.Module):
    def __init__(self, type = 1, out_dim = 1, pretrained = True, gray = False):
        super(Seresnet_Wind, self).__init__()
        if type == 1:
            name = "se_resnext50_32x4d"
        else:
            name = "se_resnext101_32x4d"
        if pretrained:
            self.extract = nn.Sequential(
                *list(pretrainedmodels.__dict__[name](num_classes=1000, pretrained="imagenet").children())[
                    :-2
                ]
            )
        else:
            self.extract = nn.Sequential(
            *list(pretrainedmodels.__dict__[name](num_classes=1000, pretrained=None).children())[
                :-2
            ]
        )
        if gray:
            self.extract[0].conv1.inchannels = 1
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Sequential(
            nn.Linear(2048, 1024),
            Swish_Module(),
            nn.Linear(1024, 512),
            Swish_Module(),
            nn.Linear(512, 256),
            Swish_Module(),
            nn.Linear(256, out_dim)
        )
        if not pretrained:
            self.extract.apply(init_weights)
            self.head.apply(init_weights)

    def forward(self, x):
        x = self.extract(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        # x = self.fea_bn(x)
        out = self.head(x)
        return out

class ResNet_Wind_LSTM(nn.Module):
    def __init__(self, gray, pretrained):
        super(CustomModels, self).__init__()
        if pretrained:
            self.extract = nn.Sequential(
                    *list(models.__dict__["resnet50"](num_classes=1000, pretrained='imagenet').children())[
                        :-1
                    ]
                )
        else:
            self.extract = nn.Sequential(
                    *list(models.__dict__["resnet50"](num_classes=1000, pretrained=None).children())[
                        :-1
                    ]
                )
        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            Swish_Module(),
            nn.Linear(512, 128),
            Swish_Module(),
            nn.LSTM(128,128)
            nn.Linear(128, 1),
        )
        if gray:
            self.extract[0].in_channels = 1
    def forward(self, x):
        x = self.extract(x)
        out = self.head(x)
        return out

        