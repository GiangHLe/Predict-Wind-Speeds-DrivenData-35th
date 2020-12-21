import argparse
import pandas as pd
from utils import ResNet_Wind, Wind_Data, get_transform, RMSELoss, write_log

import json

import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam,SGD
from tqdm import tqdm

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
    
    df_train = pd.read_csv(args.data_path + 'twenty_percent_train.csv')
    df_val = pd.read_csv(args.data_path + 'twenty_percent_val.csv')
    
    train = df_train.image_path.to_list()
    y_train = df_train.wind_speed.to_list()

    val = df_val.image_path.to_list()
    y_val = df_val.wind_speed.to_list()
    
    transform = get_transform(args.image_size)

    train_dataset = Wind_Data(
        image_list = train,
        target = y_train,
        transform = transform
    )

    val_dataset = Wind_Data(
        image_list = val,
        target = y_val,
        transform = transform
    )

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
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

    model = ResNet_Wind()
    model.to(device)

    if args.opt == 'adam':
        optimizer = Adam(model.parameters(), args.lr)
    else:
        optimizer = SGD(model.parameters(), args.lr, momentum = 0.9, nesterov= True)
    
    criterion = RMSELoss()
    
    save_path = args.save_path

    write_log(args.save_path, model, optimizer, criterion)
    
    plot_dict = {'train': list(), 'val': list()}

    with open(save_path + 'training_log.txt', 'w') as f:
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
            torch.save(model.state_dict(), 'epoch%d.pth'%(epoch))

            plot_dict['train'].append(loss)
            plot_dict['val'].append(RMSE)
            with open(save_path + 'log.json', 'w') as j:
                json.dump(plot_dict, j)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
     
    parser.add_argument('save_path', type = str, help = 'Directory where to save model and plot information')
    parser.add_argument('--data_path', type = str, default='/home/giang/Desktop/Wind_data/reproduce/', help = 'where to store data and csv file')
    parser.add_argument('--epoch', type = int, default=4, help = 'Learning rate')
    parser.add_argument('--batch-size', type = int, default=10, help = 'Batch size')
    parser.add_argument('--lr', type = int, default=2e-4, help = 'Learning rate')
    parser.add_argument('--opt', type = str, default='adam', help = 'Select optimizer')
    parser.add_argument('--image-size', type = int, default=128, help = 'Size of image input to models')
    parser.add_argument('--num-workers', type = int, default=8, help = 'Number of process in Datat Loader')

    
    parser.add_argument('--shuffle', action='store_true', help = 'shuffle while training')
    parser.add_argument('--show', action='store_true', help = 'show the result while training')
    
    
    args = parser.parse_args()

    main()
