# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 15:31:57 2022

@author: tuann
"""
from torchsummary import summary
from models.Nhom12Mang1 import *

# Initialize model
model = Net(10)
# Remove .cuda() since we want to use CPU
print("model")
print(model)

# get the number of model parameters
print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))

# Use device='cpu' in summary
summary(model, (3, 224, 224), device='cpu')
