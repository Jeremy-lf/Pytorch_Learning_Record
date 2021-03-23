# -*- coding: utf-8 -*-
"""
Created on : 2021/3/15 14:50

@author: Jeremy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)# submodule: Conv2d
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
       x = F.relu(self.conv1(x))
       return F.relu(self.conv2(x))

# add_module()
import torch.nn as nn
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.add_module("conv", nn.Conv2d(10, 20, 4))
        #self.conv = nn.Conv2d(10, 20, 4) 和上面这个增加module的方式等价
# model = Model1()
# print(model.conv)

import torch.nn as nn
class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.add_module("conv", nn.Conv2d(10, 20, 4))
        self.add_module("conv1", nn.Conv2d(20 ,10, 4))
model = Model2()

# chiledren()
# print(*(model.children()))
# for sub_module in model.children():
#     print(sub_module)

# print(*(model.modules()))

# print(*(model.named_children()))

# print(*model.parameters())

print(*model.named_parameters())

net = torch.nn.DataParallel(model, device_ids=[0, 1])
# output = net(input_var)

optimizer = torch.optim.Adam(model.parameters())

print(optimizer.state_dict())


# for input, target in dataset:
#     optimizer.zero_grad()
#     output = model(input)
#     loss = loss_fn(output, target)
#     loss.backward()
#     optimizer.step()


x = Variable(torch.Tensor([2]),requires_grad=True)

print(x)
