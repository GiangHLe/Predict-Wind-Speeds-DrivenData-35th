import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
# import torch.optim as optim
import torch_optimizer as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from dataset import get_transforms, WindDataset
from models import Seresnet_Wind
from sklearn.model_selection import train_test_split

import pickle
from utils import train_epoch, val_epoch



class Hparameter(object):
    def __init__(self):
        self.batch_size = 8
        self.lr = 0.01
        self.num_workers = 8
        self.num_epochs = 100
        self.image_size = 640

if __name__ == "__main__":
    args = Hparameter()
    device = torch.device('cuda')

    df = pd.read_csv('D:/Predict_Wind/training_set_labels.csv')
    image_id = df.image_id.to_list()
    target = df.wind_speed.to_list()

    train, val, y_train, y_val = train_test_split(image_id, target, test_size = 0.2, random_state = 42, shuffle = True)

    transforms_train, transforms_val = get_transforms(args.image_size)

    dataset_train = WindDataset(
        image_list = train, 
        target = y_train,
        test = False, 
        transform=transforms_train
        )
    dataset_valid = WindDataset(
        image_list = val,
        target = y_val,
        test = False, 
        transform=transforms_val
        )
    train_loader = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=args.batch_size, 
        sampler=RandomSampler(dataset_train), 
        num_workers=args.num_workers
        )
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=args.batch_size, 
        num_workers=args.num_workers
        )

    model = Seresnet_Wind()
    model.to(device)

    optimizer = optim.RAdam(
        model.parameters(),
        lr= 1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )

    criterion = nn.MSELoss()
    best_rmse = 12.
    rmse = []
    train_loss_overall = []
    for epoch in range(args.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        RMSE = val_epoch(model, valid_loader, criterion, device)
        rmse.append(RMSE)
        train_loss_overall.append(train_loss)
        pick = {'train': train_loss_overall, 'val':rmse}
        with open('C:/Users/giang/Desktop/predict_wind/plot.pkl', 'wb') as f:
            pickle.dump(pick, f)
        if RMSE < best_rmse:
            name = 'C:/Users/giang/Desktop/wind_model/epoch_%d_%.5f.pth'%(epoch, RMSE)
            print('Saving model...')
            torch.save(model.state_dict(), name)
    torch.save(model.state_dict(), 'C:/Users/giang/Desktop/wind_model/last_epoch_%.5f.pth'%(RMSE))
