import torch
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

from models import Seresnext_Wind, SimpleModel, ResNetFromExample, ResNetFromWeb
from dataset import WindDataset


device = torch.device('cuda')

path = './data/submission_format.csv'
# path = './data/train.csv'
df = pd.read_csv(path)

ids = df.image_id


dataset_test = WindDataset(
        image_list = ids, 
        test = True, 
        )

test_loader = DataLoader(
        dataset_test, 
        batch_size = 1024, 
        shuffle = False,
        num_workers = 12
        )

warm_up = True
train_mode = False

NAME = 'sub_warm_2'
weights_path = './weights/resnet_benhmark_change_m/epoch_9_8.13840.pth'
# model = ResNetFromExample()
# model = Seresnet_Wind(type = 1, pretrained= True, gray = False)
model = ResNetFromWeb()
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
                print(output)
                result = np.concatenate((result, output), axis = 0)

result = list((np.round(result)).astype(np.int32).flatten())
df.wind_speed = result
df.to_csv('./'+ NAME +'.csv', index = False)