# -*- coding: utf-8 -*-
"""
Created on : 2021/5/18 20:25

@author: Jeremy
"""
import paddle

from paddle.io import Dataset,DataLoader,Sampler

class MyDataset(Dataset):
    def __init__(self,mode = 'train'):
        # 步骤一：继承paddle.io.Dataset类
        super(MyDataset,self).__init__()

        #  步骤二：实现构造函数，定义数据读取方式，划分训练和测试数据集
        if mode == 'train':
            self.data = [
                [[1,2], 1],
                [[2,3], 2],
                [[3,4], 1],
                [[4,5], 2],
            ]

        else:
            self.data = [
                ['testdata1', 'label1'],
                ['testdata2', 'label2'],
                ['testdata3', 'label3'],
                ['testdata4', 'label4'],
            ]

    def __getitem__(self, item):
        # 步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        data = self.data[item][0]
        label = self.data[item][1]

        return data,label

    def __len__(self):
        # 步骤四：实现__len__方法，返回数据集总数目
        return len(self.data)


train_dataset = MyDataset(mode='train')
val_dataset = MyDataset(mode='test')

print('===========trian dataset=========')
for data,label in train_dataset:
    print(data,label)

print('==========val dataset ==========')
for data ,label in val_dataset:
    print(data,label)


# Layer类继承方式组网
class Mnist(paddle.nn.Layer):
    def __init__(self):
        super(Mnist, self).__init__()

        self.flatten = paddle.nn.Flatten()
        self.linear_1 = paddle.nn.Linear(784, 512)
        self.linear_2 = paddle.nn.Linear(512, 10)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.2)

    def forward(self, inputs):
        y = self.flatten(inputs)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_2(y)

        return y

mnist = Mnist()
