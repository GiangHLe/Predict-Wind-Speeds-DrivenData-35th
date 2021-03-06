'''
TODO: + test augment
      + test 
'''

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import argparse
import pandas as pd

from utils import (
    RMSELoss, JointLoss, JointLoss2, write_log, extract_number, \
    reset_m_batchnorm, save_pth
)

from dataset import WindDataset, get_transform

import json
import numpy as np

import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
from torch_optimizer import RAdam # optim#.RAdam as RAdam
from tqdm import tqdm

from models import *

def check_optim(optimizer):
    all_opt = {
        AdamW: 'adamw',
        Adam: 'adam',
        SGD: 'sgd',
        RAdam: 'radam'
    }
    for i in all_opt.keys():
        if isinstance(optimizer, i):
            return all_opt[i]

def warm_up(model, dataloader, optimizer, criterion, device, exp):
    model.train()
    bar = tqdm(dataloader)
    for (image, target) in bar:
        image, target = image.to(device), target.to(device)
        output = model(image)
        optimizer.zero_grad()
        if not exp:
            loss = criterion(output, target)
        else:
            loss, class_loss, reg_loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_np = loss.clone().detach().cpu().item()
        if exp:
            all_class.append(class_loss)
            all_reg.append(reg_loss)
            bar.set_description('Loss: %.4f, classify: %.4f, reg: %.4f'%(loss_np, class_loss, reg_loss))
        else:
            bar.set_description('Loss: %.4f'%(loss_np))

def train_epoch(model, dataloader, optimizer, criterion, device, exp):
    # torch.autograd.set_detect_anomaly(True)
    model.train()
    bar = tqdm(dataloader)
    all_loss = list()
    all_class = list()
    all_reg = list()
    for (image, target) in bar:
        image, target = image.to(device), target.to(device)
        output = model(image)
        optimizer.zero_grad()
        if not exp:
            loss = criterion(output, target)
        else:
            loss, class_loss, reg_loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        loss_np = loss.clone().detach().cpu().item()
        all_loss.append(loss_np)
        if exp:
            all_class.append(class_loss)
            all_reg.append(reg_loss)
            bar.set_description('Loss: %.4f, classify: %.4f, reg: %.4f'%(loss_np, class_loss, reg_loss))
        else:
            bar.set_description('Loss: %.4f'%(loss_np))
        # temp1 = output.clone().detach().cpu().numpy()
        # temp2 = target.clone().detach().cpu().numpy()
        # temp = np.concatenate((temp1,temp2), axis = 1)
        # print(temp)
    all_loss = np.mean(np.array(all_loss), axis = 0)
    if exp:
        all_class = np.mean(np.array(all_class), axis = 0)
        all_reg = np.mean(np.array(all_reg), axis = 0)
        return all_loss, all_class, all_reg
    return all_loss

def val_epoch(model, dataloader, device, exp, anchors):
    model.eval()
    bar = tqdm(dataloader)
    RMSE = 0.
    num_sample = 0.
    correct_class = 0.
    with torch.no_grad():
        for (image, target) in bar:
            image = image.to(device)
            num_sample += image.size(0)
            predict = model(image).detach().cpu().numpy()
            target = target.numpy()
            if exp:
                # convert from 4D -> 2D
                predict = np.squeeze(predict)
                # Get label
                classify_y = target[:,0].astype(np.int32)
                regression_y = target[:,1]
                # Classification results from model
                label = np.argmax(predict[:,:3], axis= 1)
                # Get correct samples from classification task to calculate accuracy and anchor to scale
                correct_class+=len(np.where(label == classify_y)[0])
                anchor = np.expand_dims(np.array([anchors[i] for i in label]),axis = 1)
                # anchor*exp(output)

                '''test'''
                regression = predict[:,3:]

                index = np.zeros(regression.shape)
                index[np.arange(label.shape[0]),label] = 1
                index = index.astype(np.bool)

                exp_scale = regression[index]
                predict = np.squeeze(anchor) * np.exp(exp_scale)

                # Ground truth
                real_anchor = np.array([anchors[i] for i in classify_y])
                target = real_anchor * np.exp(regression_y)
            RMSE += np.sum((predict - target)**2)
    RMSE = np.sqrt(RMSE/num_sample)
    return [RMSE, correct_class/num_sample] if exp else RMSE

def main():
    anchors = [30,54,95]
    shuffle = not(args.no_shuffle)
    exp = args.exp
    warm_up_epoch = 3

    # Load and process data
    
    if args.fold:
        df_train = pd.read_csv(args.data_path + 'k_fold/official_train_fold%d.csv'%(args.fold))
        df_val = pd.read_csv(args.data_path + 'k_fold/official_val_fold%d.csv'%(args.fold))
    else:
        df_train = pd.read_csv(args.data_path + 'official_train.csv')
        df_val = pd.read_csv(args.data_path + 'official_val.csv')
    
    train = df_train.image_path.to_list()
    val = df_val.image_path.to_list()
    if exp:
        y_train = df_train.anchor.to_list()
        y_val = df_val.anchor.to_list()
        reg_train_gt = df_train.exp_wind.to_list()
        reg_val_gt = df_val.exp_wind.to_list()
    else:
        y_train = df_train.wind_speed.to_list()
        y_val = df_val.wind_speed.to_list()        
    
    train_transform, val_transform = get_transform(args.image_size)

    train_dataset = WindDataset(
        image_list = train,
        target = y_train,
        exp_target = reg_train_gt if exp else None,
        transform = train_transform
    )

    val_dataset = WindDataset(
        image_list = val,
        target = y_val,
        exp_target = reg_val_gt if exp else None,
        transform = val_transform
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

    warm_loader = DataLoader(
        dataset = train_dataset,
        batch_size=args.batch_size*14,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=True
    )

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    last_epoch = 0

    # model = ResNet50_BN_idea()
    if not exp:
        model = Effnet_Wind_B7()
        # model = Effnet_Wind_B5()
    else:
        model = Effnet_Wind_B5_exp_6()
    # model = ResNetExample()
    # if not exp:
    #     model = Seresnext_Wind()
    # else:
    #     model = Seresnext_Wind_Exp()

    # Optimizer
    if args.opt == 'radam':
        optimizer = RAdam(
            model.parameters(),
            lr= args.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
        )
    elif args.opt == 'adamw':
        optimizer = AdamW(
            model.parameters(), 
            args.lr
        )

    elif args.opt == 'adam':
        optimizer = Adam(
            model.parameters(), 
            args.lr,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = SGD(
            model.parameters(), 
            args.lr, 
            momentum = 0.9, 
            nesterov= True,
            weight_decay=args.weight_decay
        )
    
    if args.weights:
        # model.load_state_dict(torch.load(args.weights))
        last_epoch = extract_number(args.weights)
        try:
            checkpoint = torch.load(args.weights)
            model.load_state_dict(checkpoint['model_state_dict'])
            if checkpoint['pre_opt'] == args.opt:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(optimizer)
        except:
            model.load_state_dict(torch.load(args.weights))
    else:
        model.apply(reset_m_batchnorm)

    model.to(device)
    
    # Loss function
    if exp:
        criterion = JointLoss2()
    else:
        criterion = RMSELoss()

    # generate log and visualization
    save_path = args.save_path

    log_cache = (
        args.batch_size,
        args.image_size,
        shuffle,
        exp
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
            plot_dict['train'] = plot_dict['train'][:last_epoch]
            plot_dict['val'] = plot_dict['val'][:last_epoch]

    # Training
    print('Start warm up')
    model.freeze_except_last()
    for epoch in range(warm_up_epoch):
        warm_up(model= model,
                dataloader= warm_loader, 
                optimizer = optimizer, 
                criterion = criterion,
                device = device,
                )
    model.unfreeze()
    with open(log_train_path, write_mode) as f:
        for epoch in range(1, args.epoch+1):
            print('Epoch:',epoch + last_epoch)
            f.write('Epoch: %d\n'%(epoch+last_epoch))
            loss = train_epoch(model = model,
                               dataloader = train_loader, 
                               optimizer = optimizer, 
                               criterion = criterion, 
                               device = device,
                               exp = exp
                            )
            RMSE = val_epoch(model = model,
                             dataloader=val_loader, 
                             device=device,
                             exp = exp,
                             anchors= anchors
                            )
            if not exp:
                f.write('Training loss: %.4f\n'%(loss))
                f.write('RMSE val: %.4f\n'%(RMSE))
                print('RMSE loss: %.4f'%(loss))
                print('RMSE val: %.4f'%(RMSE))
            else:
                loss, classify, regress = loss
                RMSE, accuracy = RMSE
                f.write('Training loss: %.4f\n'%(loss))
                f.write('Classification loss: %.4f\n'%(classify))
                f.write('Regression loss: %.4f\n'%(regress))
                f.write('Accuracy val: %.4f\n'%(accuracy))
                f.write('RMSE val: %.4f\n'%(RMSE))
                print('Training loss: %.4f'%(loss))
                print('Classification loss: %.4f'%(classify))
                print('Regression loss: %.4f'%(regress))
                print('Accuracy val: %.4f'%(accuracy))
                print('RMSE val: %.4f'%(RMSE))

            # torch.save(model.state_dict(), save_path + 'epoch%d.pth'%(epoch+last_epoch))
            save_name = save_path + 'epoch%d.pth'%(epoch+last_epoch)
            save_pth(save_name , epoch+last_epoch, model, optimizer, args.opt)

            plot_dict['train'].append(loss)
            plot_dict['val'].append(RMSE)
            with open(plot_train_path, 'w') as j:
                json.dump(plot_dict, j)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
     
    parser.add_argument('save_path', type = str, help = 'Directory where to save model and plot information')
    parser.add_argument('--weights', type = str, default = None, help = 'Weights path')
    parser.add_argument('--data_path', type = str, default='./data/', help = 'where to store data and csv file')
    parser.add_argument('--epoch', type = int, default=100, help = 'Learning rate')
    parser.add_argument('--batch-size', type = int, default=15, help = 'Batch size')
    parser.add_argument('--lr', type = float, default=3e-4, help = 'Learning rate')
    parser.add_argument('--opt', type = str, default='radam', help = 'Select optimizer')
    parser.add_argument('--image-size', type = int, default=366, help = 'Size of image input to models')
    parser.add_argument('--num-workers', type = int, default=8, help = 'Number of process in Datat Loader')
    parser.add_argument('--weight-decay', type = float, default=0., help = 'L2 Regularization')
    parser.add_argument('--fold', type = int, default=None, help = 'Fold index')

    parser.add_argument('--exp', action = 'store_true', help = 'use exponential target and clusters')
    parser.add_argument('--no-shuffle', action='store_true', help = 'shuffle while training')
    parser.add_argument('--show', action='store_true', help = 'show the result while training')
    
    
    args = parser.parse_args()

    main()
