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
from torch.optim import lr_scheduler, Adam, SGD
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from dataset import get_transforms, WindDataset
from models import Seresnet_Wind, SimpleModel, ResNetFromExample
from sklearn.model_selection import train_test_split

import pickle
from utils import train_epoch, val_epoch, RMSELoss



class Hparameter(object):
    def __init__(self):
        self.batch_size = 256
        self.lr = 2e-4
        self.num_workers = 8
        self.num_epochs = 18
        # self.image_size = 368
        self.image_size = 224
        self.save_path = './weights/resnet50-full-Switch/'

if __name__ == "__main__":
    args = Hparameter()
    device = torch.device('cuda')

    df = pd.read_csv('./data/training_set_labels.csv')
    # target = np.array(df.wind_speed).astype(np.float32)
    # max_wind = np.amax(target)
    # target = list(target / max_wind)

    image_id = df.image_id.to_list()
    target = df.wind_speed.to_list()

    # train, val, y_train, y_val = train_test_split(image_id, target, test_size = 0.2, random_state = 42, shuffle = True)

    df_train = pd.read_csv('./data/train_10first_10last.csv')
    df_val = pd.read_csv('./data/val_10first_10last.csv')
    train = df_train.image_id.to_list()
    y_train = df_train.wind_speed.to_list()

    val = df_val.image_id.to_list()
    y_val = df_val.wind_speed.to_list()
    
    transforms_train, transforms_val = get_transforms(args.image_size, gray = True)

    dataset_train = WindDataset(
        image_list = train, 
        target = y_train,
        test = False, 
        transform=transforms_train,
        # gray = True
        )
    dataset_valid = WindDataset(
        image_list = val,
        target = y_val,
        test = False, 
        transform=transforms_val,
        # gray = True,
        a = True
        )
    train_loader = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=args.batch_size, 
        # sampler=RandomSampler(dataset_train), 
        shuffle = False,
        num_workers=args.num_workers,
        drop_last=True
        )
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=True
        )

    # model = Seresnet_Wind(type = 1, pretrained= False, gray = True)
    # model = SimpleModel()
    model = ResNetFromExample()
    # print(model)
    path = './weights/resnet50-full-Switch/epoch_17_4.67716.pth'
    model.load_state_dict(torch.load(path))

    # model = ResNet_Wind_LSTM(pretrained = False, gray = True)
    model.to(device)


    # real_batch = 64
    # # acc_scale = args.batch_size / real_batch
    # acc_scale = 1
    # real_lr = args.lr*acc_scale

    # optimizer = optim.RAdam(
    #     model.parameters(),
    #     lr= args.lr,
    #     betas=(0.9, 0.999),
    #     eps=1e-8,
    #     weight_decay=0,
    # )
    optimizer = SGD(model.parameters(), lr = real_lr, momentum=0.9, nesterov= True)
    # optimizer = Adam(model.parameters(), lr = args.lr)

    criterion = RMSELoss()
    best_rmse = 30.
    rmse = []
    train_loss_overall = []
    last_epoch = 0

    for epoch in range(args.num_epochs):
        model.train()
        torch.cuda.synchronize()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        model.eval()
        torch.cuda.synchronize()
        max_wind = 1
        RMSE = val_epoch(model, valid_loader, criterion, device, max_wind)
        rmse.append(RMSE)
        train_loss_overall.append(train_loss)
        pick = {'train': train_loss_overall, 'val':rmse}
        # with open('./plot.pkl', 'wb') as f:
        #     pickle.dump(pick, f)
        # if RMSE < best_rmse or epoch%10 == 0:
        # name = args.save_path + 'epoch_%d_%.5f.pth'%(epoch + last_epoch + 1, RMSE)
        # # best_rmse = RMSE
        # print('Saving model...')
        # torch.save(model.state_dict(), name)
    # torch.save(model.state_dict(), args.save_path + 'last_epoch_%.5f.pth'%(RMSE))
