import os
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from torch.utils.data import Dataset

dataroot = '/home/giang/Desktop/Wind_data/train/'
# dataroot = 'C:/Users/Admin/Desktop/Wind_data/train/'

def get_transforms(image_size, gray = False):

    transforms_train = albumentations.Compose([
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightness(limit=0.2, p=0.75),
        albumentations.RandomContrast(limit=0.2, p=0.75),
        albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=5),
            albumentations.MedianBlur(blur_limit=5),
            albumentations.GaussianBlur(blur_limit=5),
            albumentations.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),

        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.0),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),
        ], p=0.7),

        albumentations.CLAHE(clip_limit=4.0, p=0.7),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        albumentations.Resize(image_size, image_size),
        albumentations.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7)
    ])

    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size)
        
    ])

    if not gray:
        transforms_train = albumentations.Compose([
            transforms_train,
            albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            albumentations.Normalize()
        ])
        transforms_val = albumentations.Compose([
            transforms_val,
            albumentations.Normalize()
        ])

    return transforms_train, transforms_val

class WindDataset(Dataset):
    def __init__(self, image_list, target, test, transform = None, gray = False, a = False):
        self.image_list = image_list
        self.target = target
        self.test = test
        self.transform = transform
        self.gray = gray
        self.a = a
    def __len__(self):
        if not self.a:        
            return len(self.image_list)
        else:
            return 1000
        # return len(self.image_list)
        # return 1024

    def __getitem__(self, i):
        if not self.gray:
            image = cv2.imread(dataroot + self.image_list[i] + '.jpg')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.imread(dataroot + self.image_list[i] + '.jpg', 0)
        # print(self.image_list[i])
        if self.transform:
            image = self.transform(image=image)['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)
        # image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA).astype(np.float32)

        if self.gray:
            image = np.expand_dims(image, axis = 2)
            image/=255.
        image = torch.tensor(image).float()
        image = image.permute(2,0,1)
        if self.test:
            return image
        return image, torch.tensor(self.target[i]).float()

