import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

from models import *
from dataset import WindDataset, get_transform


device = torch.device('cuda')

path = './data/submission_format.csv'
# path = './data/train.csv'
df = pd.read_csv(path)

side_df = pd.read_csv('./data/official_test.csv')
ids = side_df.image_path
# ids = df.image_id

# prefix = 'C:/Users/Admin/Desktop/Wind_data/test/'
# prefix = 'D:/Predict_Wind/test/'

# ids = [prefix + str(i) + '.jpg' for i in ids]
batch_size = 150

_, transform = get_transform(366)

dataset_test = WindDataset(
        image_list = ids,
        transform = transform, 
        test = True
        )

shuffle_loader = DataLoader(
        dataset_test, 
        batch_size = batch_size, 
        shuffle = True,
        num_workers = 8
        )

test_loader = DataLoader(
        dataset_test, 
        batch_size = batch_size, 
        shuffle = False,
        num_workers = 8
        )

warm_up = True
train_mode = False
exp = False
NAME = 'EffB7_fold5'
weights_path = './weights/EffiNetB7_fold/fold5/epoch32.pth'
# model = ResNetFromExample()
# model = Seresnet_Wind(type = 1, pretrained= True, gray = False)
# model = Seresnext_Wind_Exp()
# model = Effnet_Wind_B5_exp()
model = Effnet_Wind_B7()
# model = ResNet50_BN_idea()
# model.load_state_dict(torch.load(weights_path))
checkpoint = torch.load(weights_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

from tqdm import tqdm

# bar = tqdm(test_loader)

result = np.zeros((0,1), dtype = np.float32)

with torch.no_grad():
        if warm_up:
                print('Warm up....')
                model.train()
                for k,image in enumerate(shuffle_loader):
                        image = image.to(device)
                        outpt = model(image)
                        if k == 2048//batch_size:
                                break
                model.eval()
        if train_mode:
                model.train()
        for image in tqdm(test_loader):
                image = image.to(device)
                output = model(image).detach().cpu().numpy()
                # print(output)
                # print(output)
                if exp:
                        predict = np.squeeze(output)
                        anchors = [30, 54, 95]
                        label = np.argmax(predict[:,:3], axis= 1)
                        anchor = np.array([anchors[i] for i in label])
                        exp_scale = predict[:,3]
                        output = anchor * np.exp(exp_scale)
                        # print(output.shape,anchor.shape,exp_scale.shape)
                        output = np.expand_dims(output, axis=1)
                result = np.concatenate((result, output), axis = 0)

result = list((np.round(result)).astype(np.int32).flatten())
df.wind_speed = result
df.to_csv('./'+ NAME +'.csv', index = False)