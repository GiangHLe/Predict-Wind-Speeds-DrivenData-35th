import torch
from tqdm import tqdm
import numpy as np
def train_epoch(model, loader, optimizer, criterion, device):

    model.train()
    train_loss = []
    bar = tqdm(loader)
    running_loss = 0
    for i, (image, target) in enumerate(bar):
        image, target = image.to(device), target.to(device)
        b_size = image.size()[0]
        output = model(image)        
        
        loss = criterion(output, target)
        optimizer.step()

        loss_np = loss.item()
        train_loss.append(loss_np)
        # running_loss += loss_np*b_size
        bar.set_description('loss: %.5f' % (loss_np))
    train_loss = np.mean(train_loss)
    bar.set_description('running loss: %.5f' % (train_loss))
    return train_loss

def val_epoch(model, loader, criterion, device):

    model.eval()
    RMSE = 0
    num_sample = 0
    bar = tqdm(loader)
    with torch.no_grad():
        for (image,target) in bar:
            image, target = image.to(device), target.to(device)
            num_sample += image.size()[0]
            output = model(image).detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            RMSE += np.sum((output - target)**2)
        RMSE/=num_sample
        bar.set_description('RMSE: %.5f' % (RMSE))
    return RMSE

