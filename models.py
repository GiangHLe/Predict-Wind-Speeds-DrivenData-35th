import torch
import torch.nn as nn
import pretrainedmodels
from utils import init_weights
from torchvision import models

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


class Seresnext_Wind(nn.Module):
    def __init__(self, type = 1, out_dim = 1, pretrained = True, gray = False):
        super(Seresnext_Wind, self).__init__()
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
            # print(self.extract[0].conv1)
            self.extract[0].conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Dropout(p = 0.2),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.Dropout(p = 0.2),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(128),
            nn.Linear(128, 32),
            nn.Dropout(p = 0.2),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(32),
            nn.Linear(32, out_dim)
        )
        # self.fea_bn = nn.BatchNorm1d(2048)
        # self.fea_bn.bias.requires_grad_(False)

        if not pretrained:
            self.extract.apply(init_weights)
            self.head.apply(init_weights)
            # self.fea_bn.apply(init_weights)
    
    def forward(self, x):
        x = self.extract(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        # x = self.fea_bn(x)
        out = self.head(x)
        return out

class Seresnext_Wind_DenseShallow_Swish(nn.Module):
    def __init__(self, type = 1, out_dim = 1, pretrained = True, gray = False):
        super(Seresnext_Wind_DenseShallow_Swish, self).__init__()
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
            # print(self.extract[0].conv1)
            self.extract[0].conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Dropout(p = 0.2),
            Swish_Module(),
            # nn.BatchNorm1d(512, momentum= 0.3),
            nn.Linear(512, 128),
            nn.Dropout(p = 0.2),
            Swish_Module(),
            nn.Linear(128, 1)
        )
        # self.fea_bn = nn.BatchNorm1d(2048)
        # self.head[3].bias.requires_grad_(False)

        if not pretrained:
            self.extract.apply(init_weights)
            self.head.apply(init_weights)
            # self.fea_bn.apply(init_weights)
    
    def forward(self, x):
        x = self.extract(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        # x = self.fea_bn(x)
        out = self.head(x)
        return out

class Seresnext_Wind_Conv2d_Swish(nn.Module):
    def __init__(self, type = 1, out_dim = 1, pretrained = True, gray = False):
        super(Seresnext_Wind_Conv2d_Swish, self).__init__()
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
            # print(self.extract[0].conv1)
            self.extract[0].conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.middle = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.2, affine=True, track_running_stats=True),
            nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.2, affine=True, track_running_stats=True),
            # Swish_Module()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.conv_out = nn.Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        # self.fea_bn = nn.BatchNorm1d(2048)
        # self.head[3].bias.requires_grad_(False)

        if not pretrained:
            self.extract.apply( )
            # self.head.apply(init_weights)
            self.conv_out.apply(init_weights)
            self.middle.apply(init_weights)
            # self.fea_bn.apply(init_weights)
    
    def forward(self, x):
        x = self.extract(x)
        x = self.middle(x)
        x = self.avg_pool(x)
        # print(x.size())
        # x = x.view(x.size(0), -1)
        # x = self.fea_bn(x)
        out = self.conv_out(x)
        return out.view(out.size(0), -1)

class ResNet_Wind_LSTM(nn.Module):
    def __init__(self, gray, pretrained):
        super(ResNet_Wind_LSTM, self).__init__()
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
            nn.Dropout(p = 0.5),
            Swish_Module(),
            nn.Linear(512, 128),
            nn.Dropout(p = 0.5),
            Swish_Module(),
            nn.LSTM(128,128),
            nn.Linear(128, 1)
        )
        if gray:
            self.extract[0].in_channels = 1
        if not pretrained:
            print('Init weight...')
            self.extract.apply(init_weights)
            self.head.apply(init_weights)
    def forward(self, x):
        x = self.extract(x)
        out = self.head(x)
        return out
    

class ResNetFromExample(nn.Module):
    def __init__(self, pretrained = True):
        super(ResNetFromExample, self).__init__()
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
            nn.Linear(2048, 50),
            nn.Dropout(p = 0.1),
            nn.ReLU(),
            nn.Linear(50, 1),
        )
    def forward(self, x):
        x = self.extract(x)
        x = x.view(x.size(0), -1)
        out = self.head(x)
        return out

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.extract = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 32),
            nn.Dropout(p = 0.2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
            nn.ReLU()            
        )
    def forward(self, x):
        x = self.extract(x)
        x = x.view(x.size(0), -1)
        out = self.head(x)
        return out