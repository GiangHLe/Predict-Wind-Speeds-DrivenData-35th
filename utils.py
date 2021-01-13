import torch
import torch.nn as nn
from torch.utils.data import Dataset

import os, sys
import numpy as np

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
    def forward(self, x, y):
        return torch.sqrt(self.mse(x, y))

class JointLoss(nn.Module):
    def __init__(self):
        super(JointLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
    def forward(self, x, y):
        # Joint the loss from classification part and regression part,
        # only update for regression when the classify correct. 
        classify = x[:,:3].squeeze()
        regression = x[:,3:].squeeze()
        y_class = y[:,0].type(torch.LongTensor).to(y.device)
        y_reg  = y[:,1]
        pred = torch.argmax(classify, dim = 1).to(y_class.device)
        mask = (pred==y_class).flatten()
        
        classify_loss = 5*self.ce(classify, y_class) # add coordinate, since the classification part is the key of this method
        
        # regression[torch.logical_not(mask)] = 0.0
        # y_reg[torch.logical_not(mask)] = 0.0
        regression_loss = self.mse(regression[mask], y_reg[mask])
        # regression_loss = self.mse(regression, y_reg)
    
        return classify_loss + regression_loss

sigmoid = nn.Sigmoid()

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)

class HiddenPrints: # Block stdout in print function
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def write_log(save_path, model, optimizer, criterion, cache): # dump training information to txt file 
    with HiddenPrints():
        with open(save_path + 'model_info.txt', 'w') as f:
            print(model, file = f)
        with open(save_path + 'training_info.txt', 'w') as f:
            f.write('Optimizer: \n')
            print(optimizer, file = f)
            f.write('Loss Function: \n')
            print(criterion, file = f)

            batch_size, image_size, shuffle, exp = cache
            f.write('Batch size: %d\n'%(batch_size))
            f.write('Input image size: %d\n'%(image_size))
            f.write('Data shuffle: '+str(shuffle) + '\n' )
            f.write('Exponential target: '+str(exp) + '\n' )

def extract_number(path):
    if not os.path.exists(path):
        raise Exception('Path is not exists')
    name, extension = os.path.splitext(path)
    if extension != '.pth':
        raise Exception('Only accept file with pth extension')
    real_name = name.split('/')[-1]
    return int(real_name.split('epoch')[-1])

def fifty_m_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.momentum = 0.5

def reset_m_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.running_mean = torch.zeros(m.running_mean.size())
        m.running_var = torch.zeros(m.running_var.size())
        m.momentum = 0.5

def back_normal_m_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.momentum = 0.1

def save_pth(path, epoch, model, optimizer, type_opt):
    torch.save({
        'epoch': epoch,
        'pre_opt': type_opt,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
