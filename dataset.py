import torch
from torch.utils.data import Dataset

import cv2
import numpy as np
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

def get_transform(image_size):
    transform = albumentations.Compose([
        albumentations.CenterCrop(image_size, image_size),
        albumentations.Normalize(),
        ToTensorV2() # always use V2 follow this answer: https://albumentations.ai/docs/faq/#which-transformation-should-i-use-to-convert-a-numpy-array-with-an-image-or-a-mask-to-a-pytorch-tensor-totensor-or-totensorv2
    ]
    )
    return transform

class WindDataset(Dataset):
    def __init__(self, image_list, target = None, transform = None, test = False):
        self.image_list = image_list
        self.target = target
        self.transform = transform
        self.test = test

    def __len__(self):
        # return int(0.2*len(self.image_list))
        return len(self.image_list)

    def __getitem__(self, i):
        # read by PIL and transform with pytorch in original 
        image = cv2.imread(self.image_list[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        if self.transform:
            image = self.transform(image = image)['image']
        if self.test:
            return image
        return image, torch.Tensor([self.target[i]]).float()
