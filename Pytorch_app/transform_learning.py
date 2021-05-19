# -*- coding: utf-8 -*-
"""
Created on : 2021/1/19 12:43

@author: Jeremy
"""
'''
1、微调**Convnet**：使用预训练的网络(如在 imagenet 1000 上训练而来的网络)来初始化自己
的网络，而不是随机初始化。其他的训练步骤不变。

2、将**Convnet**看成固定的特征提取器:首先固定ConvNet除了最后的全连接层外的其他所有
层。最后的全连接层被替换成一个新的随机 初始化的层，只有这个新的层会被训练[只有这层
参数会在反向传播时更新]
'''


import torchvision
import torchvision.models as models
import torch
import torch.nn as nn
model_conv = torchvision.models.resnet18(pretrained=False)

for param in model_conv.parameters():
    param.requires_grad = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = models.resnet18(pretrained=False)


model_ft.layer4 = nn.Linear(256,512)

model_ft.layer3[1] = nn.Sequential(nn.Conv2d(256,256,3,1,1),nn.Linear(256,256))
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

print(model_ft)

model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
# 观察所有参数都正在优化
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)



