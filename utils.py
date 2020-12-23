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

            batch_size, image_size, shuffle = cache
            f.write('Batch size: %d\n'%(batch_size))
            f.write('Input image size: %d\n'%(image_size))
            f.write('Data shuffle: '+str(shuffle) + '\n' )
