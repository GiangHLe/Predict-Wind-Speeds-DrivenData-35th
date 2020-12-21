import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models

import os, sys
import cv2
import numpy as np
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
class ResNet_Wind(nn.Module):
    def __init__(self):
        super(ResNet_Wind, self).__init__()
        self.model = models.resnet152(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(100,1)
        )
    def forward(self, x):
        return self.model(x)

def get_transform(image_size):
    transform = albumentations.Compose([
        albumentations.CenterCrop(image_size, image_size),
        albumentations.Normalize(),
        ToTensorV2() # always use V2 follow this answer: https://albumentations.ai/docs/faq/#which-transformation-should-i-use-to-convert-a-numpy-array-with-an-image-or-a-mask-to-a-pytorch-tensor-totensor-or-totensorv2
    ]
    )
    return transform

class Wind_Data(Dataset):
    def __init__(self, image_list, target, transform = None, test = False):
        self.image_list = image_list
        self.target = target
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        # read by PIL and transform with pytorch in original 
        image = cv2.imread(self.image_list[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        if self.transform:
            image = self.transform(image = image)['image']
        if self.test:
            return image
        # print(image.size(), torch.Tensor(self.target[i]).float())
        return image, torch.Tensor([self.target[i]]).float()

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
    def forward(self, x):
        return torch.sqrt(self.mse(x))

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def write_log(save_path, model, optimizer, criterion):
    with HiddenPrints():
        with open(save_path + 'model_info.txt', 'w') as f:
            print(model, file = f)
        with open(save_path + 'training_info.txt', 'w') as f:
            f.write('Optimizer: \n')
            print(optimizer, file = f)
            f.write('Loss Function: \n')
            print(criterion, file = f)
