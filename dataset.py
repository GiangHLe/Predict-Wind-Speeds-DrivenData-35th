import torch
from torch.utils.data import Dataset

import cv2
import numpy as np
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

def get_transform(image_size, base_size = 366):
    if image_size > base_size:
        resize = albumentations.Resize(image_size, image_size)
    else:
        resize = albumentations.CenterCrop(image_size, image_size)
    
    train_transform = albumentations.Compose([
        albumentations.Transpose(p=0.3),
        albumentations.VerticalFlip(p=0.3),
        albumentations.HorizontalFlip(p=0.3),
        albumentations.CLAHE(p=0.3),
        albumentations.Equalize(p=0.3),
        albumentations.OneOf([
            albumentations.GaussianBlur(blur_limit=5),
            albumentations.GaussNoise(var_limit = (5, 30)),
            albumentations.MedianBlur(blur_limit = 5)
        ], p = 0.5),
        resize,
        albumentations.Cutout(max_h_size = int(image_size * 0.2), max_w_size = int(image_size * 0.2), num_holes = 1, p =0.3),
        albumentations.Normalize(), 
        ToTensorV2()
    ])
    val_transform = albumentations.Compose([
        resize,
        albumentations.Normalize(),
        ToTensorV2() # always use V2 follow this answer: https://albumentations.ai/docs/faq/#which-transformation-should-i-use-to-convert-a-numpy-array-with-an-image-or-a-mask-to-a-pytorch-tensor-totensor-or-totensorv2
    ]
    )
    return train_transform, val_transform

class WindDataset(Dataset):
    def __init__(self, image_list, target = None, transform = None, test = False):
        self.image_list = image_list
        self.target = target
        self.transform = transform
        self.test = test

    def __len__(self):
        # return int(0.1*len(self.image_list))
        return len(self.image_list)
        # return 1024

    def __getitem__(self, i):
        # read by PIL and transform with pytorch in original 
        image = cv2.imread(self.image_list[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        if self.transform:
            image = self.transform(image = image)['image']
        if self.test:
            return image
        return image, torch.Tensor([self.target[i]]).float()
        
