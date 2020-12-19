import torch
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

from models import Seresnet_Wind, SimpleModel, ResNetFromExample
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
        batch_size = 256, 
        shuffle = False,
        num_workers = 12
        )

NAME = 'sub3'
weights_path = './weights/resnet50-full-Switch/epoch_30_4.63914.pth'
model = ResNetFromExample()
model.load_state_dict(torch.load(weights_path))
model.to(device)
# model.eval()
model.train()

from tqdm import tqdm

bar = tqdm(test_loader)

result = np.zeros((0,1), dtype = np.float32)

# with torch.no_grad():
for image in bar:
        image = image.to(device)
        output = model(image).detach().cpu().numpy()
        print(output)
        result = np.concatenate((result, output), axis = 0)

result = list((np.round(result)).astype(np.int32).flatten())
df.wind_speed = result
df.to_csv('./'+ NAME +'.csv', index = False)