import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import argparse
import pandas as pd

from utils import RMSELoss, write_log
from dataset import WindDataset, get_transform

import json
import numpy as np

import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam,SGD
import torch_optimizer as optim
from tqdm import tqdm

from models import *

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    bar = tqdm(dataloader)
    all_loss = list()
    for (image, target) in bar:
        image, target = image.to(device), target.to(device)
        output = model(image)
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        loss_np = loss.detach().cpu().item()
        all_loss.append(loss_np)
        bar.set_description('Loss %.4f'%(loss_np))

        # temp1 = output.clone().detach().cpu().numpy()
        # temp2 = target.clone().detach().cpu().numpy()
        # temp = np.concatenate((temp1,temp2), axis = 1)
        # print(temp)
    return np.mean(all_loss)

def val_epoch(model, dataloader, device):
    model.eval()
    bar = tqdm(dataloader)
    RMSE = 0.
    num_sample = 0.
    with torch.no_grad():
        for (image, target) in bar:
            image = image.to(device)
            num_sample += image.size(0)
            output = model(image).detach().cpu().numpy()
            target = target.numpy()
            RMSE += np.sum((output - target)**2)
    return np.sqrt(RMSE/num_sample)


def main():

    shuffle = not(args.no_shuffle)
    
    df_train = pd.read_csv(args.data_path + 'official_train.csv')
    df_val = pd.read_csv(args.data_path + 'official_val.csv')
    
    train = df_train.image_path.to_list()
    y_train = df_train.wind_speed.to_list()

    val = df_val.image_path.to_list()
    y_val = df_val.wind_speed.to_list()
    
    transform = get_transform(args.image_size)

    train_dataset = WindDataset(
        image_list = train,
        target = y_train,
        transform = transform
    )

    val_dataset = WindDataset(
        image_list = val,
        target = y_val,
        transform = transform
    )

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = ResNetExample()
    model = Seresnext_Wind()
    model.to(device)

    if args.opt == 'radam':
        optimizer = optim.RAdam(
            model.parameters(),
            lr= args.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
        )
    elif args.opt == 'adam':
        optimizer = Adam(model.parameters(), args.lr)
    else:
        optimizer = SGD(model.parameters(), args.lr, momentum = 0.9, nesterov= True)
    
    criterion = RMSELoss()
    
    save_path = args.save_path

    log_cache = (
        args.batch_size,
        args.image_size,
        shuffle
    )

    write_log(
        args.save_path, 
        model, 
        optimizer, 
        criterion,
        log_cache
    )
    
    plot_dict = {'train': list(), 'val': list()}
    
    log_train_path = save_path + 'training_log.txt'
    plot_train_path = save_path + 'log.json'

    write_mode = 'w'

    if os.path.exists(log_train_path) and os.path.exists(plot_train_path):
        write_mode = 'a'
        with open(plot_train_path, 'r') as j:
            plot_dict = json.load(j)

    with open(log_train_path, write_mode) as f:
        for epoch in range(1, args.epoch+1):
            f.write('Epoch: %d\n'%(epoch))
            loss = train_epoch(model = model,
                               dataloader = train_loader, 
                               optimizer = optimizer, 
                               criterion = criterion, 
                               device = device)
            RMSE = val_epoch(model = model,
                             dataloader=val_loader, 
                             device=device)
            f.write('Training loss: %.4f\n'%(loss))
            f.write('RMSE val: %.4f\n'%(RMSE))
            print('RMSE val: %.4f'%(RMSE))
            torch.save(model.state_dict(), save_path + 'epoch%d.pth'%(epoch))

            plot_dict['train'].append(loss)
            plot_dict['val'].append(RMSE)
            with open(plot_train_path, 'w') as j:
                json.dump(plot_dict, j)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
     
    parser.add_argument('save_path', type = str, help = 'Directory where to save model and plot information')
    parser.add_argument('--data_path', type = str, default='./data/', help = 'where to store data and csv file')
    parser.add_argument('--epoch', type = int, default=20, help = 'Learning rate')
    parser.add_argument('--batch-size', type = int, default=95, help = 'Batch size')
    parser.add_argument('--lr', type = int, default=2e-3, help = 'Learning rate')
    parser.add_argument('--opt', type = str, default='radam', help = 'Select optimizer')
    parser.add_argument('--image-size', type = int, default=224, help = 'Size of image input to models')
    parser.add_argument('--num-workers', type = int, default=8, help = 'Number of process in Datat Loader')
    
    parser.add_argument('--no-shuffle', action='store_true', help = 'shuffle while training')
    parser.add_argument('--show', action='store_true', help = 'show the result while training')
    
    
    args = parser.parse_args()

    main()
