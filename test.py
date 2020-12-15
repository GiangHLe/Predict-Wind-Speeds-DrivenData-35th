import pretrainedmodels
import torch.nn as nn

model_body = nn.Sequential(
                *list(pretrainedmodels.__dict__["se_resnext50_32x4d"](num_classes=1000, pretrained="imagenet").children())[
                    :-2
                ]
            )