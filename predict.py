import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

from models import Seresnext_Wind
from dataset import WindDataset, get_transform


device = torch.device('cuda')

path = './data/submission_format.csv'
# path = './data/train.csv'
df = pd.read_csv(path)

ids = df.image_id

# prefix = 'C:/Users/Admin/Desktop/Wind_data/test/'
prefix = 'D:/Predict_Wind/test/'

ids = [prefix + str(i) + '.jpg' for i in ids]

transform = get_transform(224)

dataset_test = WindDataset(
        image_list = ids,
        transform = transform, 
        test = True
        )

test_loader = DataLoader(
        dataset_test, 
        batch_size = 1024, 
        shuffle = False,
        num_workers = 0
        )

warm_up = True
train_mode = False

NAME = 'sub_reset_sgd_l2_warm_more'
weights_path = './../epoch39.pth'
# model = ResNetFromExample()
# model = Seresnet_Wind(type = 1, pretrained= True, gray = False)
model = Seresnext_Wind()
model.load_state_dict(torch.load(weights_path))
model.to(device)
model.eval()

from tqdm import tqdm

bar = tqdm(test_loader)

result = np.zeros((0,1), dtype = np.float32)

with torch.no_grad():
        if warm_up:
                print('Warm up....')
                model.train()
                for k,image in enumerate(test_loader):
                        image = image.to(device)
                        outpt = model(image)
                        if k == 2:
                                break
                model.eval()
        if train_mode:
                model.train()
        for image in bar:
                image = image.to(device)
                output = model(image).detach().cpu().numpy()
                # print(output)
                result = np.concatenate((result, output), axis = 0)

result = list((np.round(result)).astype(np.int32).flatten())
df.wind_speed = result
df.to_csv('./'+ NAME +'.csv', index = False)