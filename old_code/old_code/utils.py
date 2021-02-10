import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import time

def train_epoch(model, loader, optimizer, criterion, device):

    model.train()
    train_loss = []
    bar = tqdm(loader)
    running_loss = 0
    optimizer.zero_grad()

    for i, (image, target) in enumerate(bar):
    
        image, target = image.to(device), target.to(device)
        b_size = image.size()[0]
        output = model(image)    
        loss = criterion(output, target)        
        optimizer.zero_grad(set_to_none = True)
        loss.backward()
        optimizer.step()

        loss_np = loss.detach().cpu().item()
        print(loss_np)
        train_loss.append(loss_np)
        # print(temp)
        # if i%4==0:
        # temp_image = image.clone().detach().cpu()
        # temp_target = target.clone().detach().cpu()
        # print(type(temp_image), type(temp_target))
        # print(temp_image.size(), temp_target.size())
        # print(temp_image, temp_target)
        import time
        time.sleep(3)
        # temp1 = output.detach().cpu().numpy()# * 185.
        # temp2 = target.detach().cpu().numpy()# * 185.
        # temp2 = np.expand_dims(temp2, axis = 1)
        
        # mean = 50.344008426206635
        # temp1 = t2wind(mean, temp1)
        # temp2 = t2wind(mean, temp2)

        # temp = np.concatenate((temp1,temp2), axis = 1)
        # print(temp)

        bar.set_description('loss: %.5f' % (loss_np))
    
    # debug
    # previous = model.head[12].weight.clone()
    # previous = model.head[-2].weight.clone()
    # for i, (image, target) in enumerate(bar):
    #     image, target = image.to(device), target.to(device)
    #     b_size = image.size()[0]
    #     break

    # while(True):
    #     output = model(image)    
    #     loss = criterion(output, target)
    #     # now = model.model[0].weight
    #     now = model.head[-2].weight
    #     print('='*10)
    #     print(previous,now)
    #     if torch.sum(now-previous) == 0:
    #         print('yes')
    #     # print(model.head[11].running_mean)
    #     # print(model.head[12].weight.grad)
    #     optimizer.zero_grad()
    #     # print(model.head[12].weight.grad)
    #     loss.backward()
    #     optimizer.step()

    #     loss_np = loss.item()
    #     train_loss.append(loss_np)
    #     temp1 = output.detach().cpu().numpy() * 185.
    #     temp2 = target.detach().cpu().numpy() * 185.
    #     temp2 = np.expand_dims(temp2, axis = 1)
    #     temp = np.concatenate((temp1,temp2), axis = 1)
    #     print(temp)
        # print(loss_np)
        # bar.set_description('loss: %.5f' % (loss_np))
    
    
    
    train_loss = np.mean(train_loss)
    print('running loss: %.5f' % (train_loss))
    return train_loss

def val_epoch(model, loader, criterion, device, max_value, mean):

    model.eval()
    RMSE = 0
    num_sample = 0
    bar = tqdm(loader)
    with torch.no_grad():
        for (image,target) in bar:
            image, target = image.to(device), target.to(device)
            num_sample += image.size()[0]
            output = model(image).detach().cpu().numpy()
            output*=max_value
            # target = np.expand_dims(target.detach().cpu().numpy(), axis = 1)
            target = target.detach().cpu().numpy()
            target*=max_value
            if mean is not None:
                output = t2wind(mean, output)
                target = t2wind(mean, target)
            RMSE += np.sum((output - target)**2)
            
            temp = np.concatenate((output,target), axis = 1)
            print(temp)
        RMSE/=num_sample
        RMSE = np.sqrt(RMSE)
        
        print('RMSE: %.5f' % (RMSE))
    return RMSE

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()

def config_momentum(m):
    if isinstance(m, nn.BatchNorm2d):
        m.momentum = 0.4
            
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        return torch.sqrt(self.mse(output, target))

def t2wind(mean, t):
    return mean * np.exp(t)