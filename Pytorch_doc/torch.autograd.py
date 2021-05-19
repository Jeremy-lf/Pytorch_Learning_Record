# -*- coding: utf-8 -*-
"""
Created on : 2021/3/16 15:21

@author: Jeremy
"""
import torch
from torch.autograd import Variable

# x = Variable(torch.ones(2, 2), requires_grad=True)
# print(x)  # 其实查询的是x.data,是个tensor

'''
Varibale包含三个属性：

data：存储了Tensor，是本体的数据
grad：保存了data的梯度，本事是个Variable而非Tensor，与data形状一致
grad_fn：指向Function对象，用于反向传播的梯度计算之用
'''

from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
#  cap = dset.CocoCaptions(root = 'dir where images are',
#                         annFile = 'json annotation file',
#                         transform=transforms.ToTensor())
#
# minist = dset.MNIST()
# # print('Number of samples: ', len(cap))
# img, target = cap[3] # load 4th sample

# print("Image Size: ", img.size())
# print(target)


def f(x):
    """x^2 * e^x"""
    y = x ** 2 * torch.exp(x)
    return y


def gradf(x):
    """2*x*e^x + x^2*e^x"""
    dx = 2 * x * torch.exp(x) + x ** 2 * torch.exp(x)
    return dx


x = Variable(torch.randn(3, 4), requires_grad=True)
y = f(x)
y.backward(torch.ones(y.size()))
# y.backward()
print(x.grad)

print(gradf(x))  # 两个结果一样