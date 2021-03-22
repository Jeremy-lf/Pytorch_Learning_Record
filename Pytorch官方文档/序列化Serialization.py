# -*- coding: utf-8 -*-
"""
Created on : 2021/3/15 13:29

@author: Jeremy
"""
import torch

torch.load('tensors.pt')
# Load all tensors onto the CPU
torch.load('tensors.pt', map_location=lambda storage, loc: storage)
# Map tensors from GPU 1 to GPU 0
torch.load('tensors.pt', map_location={'cuda:1':'cuda:0'})