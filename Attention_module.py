# -*- coding: utf-8 -*-
"""
Created on : 2021/4/14 20:36

@author: Jeremy
"""
import torch.nn as nn
import torch


'''
SENet
'''
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # bxcx1x1 == bxc
        y = self.avg_pool(x).view(b, c)
        # bxcx1x1
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)

# input = torch.randn(2,224,224,224)
# model = SELayer(224)
# print(model)
# output = model(input)
# print(output.shape)

'''
CBAM 通道与空间的注意力机制
'''


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1) # Bx2xhxw
        x = self.conv(x) # Bx1xhxw
        return x*self.sigmoid(x) # Bxcxhxw  * Bx1xhxw


import torch.nn as nn
from collections import OrderedDict


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv_block = torch.nn.Sequential()
        self.conv_block.add_module("conv1", torch.nn.Conv2d(3, 32, 3, 1, 1))
        self.conv_block.add_module("relu1", torch.nn.ReLU())
        self.conv_block.add_module("pool1", torch.nn.MaxPool2d(2))

        self.dense_block = torch.nn.Sequential()
        self.dense_block.add_module("dense1", torch.nn.Linear(32 * 3 * 3, 128))
        self.dense_block.add_module("relu2", torch.nn.ReLU())
        self.dense_block.add_module("dense2", torch.nn.Linear(128, 10))

    def forward(self, x):
        conv_out = self.conv_block(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense_block(res)
        return out


model = MyNet()
print(model.state_dict().items())
# for i,dic in model.named_parameters():
#     print(dic)

