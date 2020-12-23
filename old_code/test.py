# import pretrainedmodels
# import torch.nn as nn

# model_body = nn.Sequential(
#                 *list(pretrainedmodels.__dict__["se_resnext50_32x4d"](num_classes=1000, pretrained="imagenet").children())[
#                     :-2
#                 ]
#             )

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
from models import Seresnet_Wind, SimpleModel
from sklearn.model_selection import train_test_split

import pickle
from utils import train_epoch, val_epoch



class Hparameter(object):
    def __init__(self):
        self.batch_size = 256
        self.lr = 1e-2
        self.num_workers = 8
        self.num_epochs = 100
        # self.image_size = 368
        self.image_size = 224
        self.save_path = './weights/'

if __name__ == "__main__":
    args = Hparameter()
    device = torch.device('cuda')

    df = pd.read_csv('./data/training_set_labels.csv')
    # target = np.array(df.wind_speed).astype(np.float32)

    image_id = df.image_id.to_list()
    target = df.wind_speed.to_list()

    train, val, y_train, y_val = train_test_split(image_id, target, test_size = 0.2, random_state = 42, shuffle = True)

    transforms_train, transforms_val = get_transforms(args.image_size, gray = True)

    dataset_train = WindDataset(
        image_list = train, 
        target = y_train,
        test = False, 
        transform=transforms_train,
        gray = True
        )
    dataset_valid = WindDataset(
        image_list = val,
        target = y_val,
        test = False, 
        transform=transforms_val,
        gray = True,
        a = True
        )
    train_loader = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=args.batch_size, 
        shuffle = False,
        num_workers=args.num_workers
        )
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=args.batch_size*2, 
        num_workers=args.num_workers,
        shuffle=False
        )

    model = SimpleModel()
    model.to(device)

    optimizer = SGD(model.parameters(), lr = args.lr , momentum=0.9, nesterov= True)

    criterion = nn.MSELoss()
    best_rmse = 12.

    rmse = []
    train_loss_overall = []

    for epoch in range(args.num_epochs):
        model.train()
        torch.cuda.synchronize()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        model.eval()
        torch.cuda.synchronize()
        RMSE = val_epoch(model, valid_loader, criterion, device, max_wind)
        rmse.append(RMSE)
        train_loss_overall.append(train_loss)
        pick = {'train': train_loss_overall, 'val':rmse}
        with open('./plot.pkl', 'wb') as f:
            pickle.dump(pick, f)
        if RMSE < best_rmse:
            name = args.save_path + 'epoch_%d_%.5f.pth'%(epoch, RMSE)
            print('Saving model...')
            torch.save(model.state_dict(), name)
    torch.save(model.state_dict(), args.save_path + 'last_epoch_%.5f.pth'%(RMSE))
