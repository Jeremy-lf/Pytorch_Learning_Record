# -*- coding: utf-8 -*-
"""
Created on : 2021/3/18 13:31

@author: Jeremy
"""
import torch
import numpy as np


'''
产生

y = torch.empty(3, dtype=torch.long).random_(5)

y = torch.Tensor(2,3).random_(10)

y = torch.randn(3,4).random_(10)

查看张量的类型
y.type()
y.dtype

long(),int(),float() 实现类型的转化
'''


def one_hot(y):
    '''
    y: (N)的一维tensor，值为每个样本的类别
    out:
        y_onehot: 转换为one_hot 编码格式
    '''
    y = y.view(-1, 1)

    # y_onehot = torch.FloatTensor(3, 5)
    # y_onehot.zero_()

    y_onehot = torch.zeros(3,5)  # 等价于上面
    y_onehot.scatter_(1, y, 1)
    return y_onehot

y = torch.empty(3, dtype=torch.long).random_(5)
# print(y)
res = one_hot(y)
h = torch.argmax(res,dim=1)
_,h1 = res.max(dim=1)
print("h",h)
print("h1",h1)
print(res)


class_num = 10
batch_size = 4
label = torch.LongTensor(batch_size, 1).random_() % class_num

print(label)
one_hot = torch.zeros(batch_size, class_num).scatter_(1, label, 1)