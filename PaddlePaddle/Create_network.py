# -*- coding: utf-8 -*-
"""
Created on : 2021/5/19 15:36

@author: Jeremy
"""
import paddle
from paddle.io import Dataset,DataLoader

BATCH_SIZE = 64
BATCH_NUM = 20

IMAGE_SIZE = (28,28)
CLASS_NUM = 10

class MyDataset(Dataset):
    def __init__(self,num_samples):
        super(MyDataset,self).__init__()
        self.num_samples = num_samples

    def __getitem__(self, item):
        data = paddle.uniform(IMAGE_SIZE,dtype='float32')
        label = paddle.randint(0,CLASS_NUM-1,dtype='int64')

        return data,label

    def __len__(self):
        return self.num_samples

custom_dataset = MyDataset(BATCH_SIZE*BATCH_NUM)
print('=============custom dataset=============')
for data, label in custom_dataset:
    print(data.shape, label.shape)
    break

train_loader = paddle.io.DataLoader(custom_dataset,batch_size=BATCH_SIZE,shuffle=True)

for batch_id,data in enumerate(train_loader):
    x = data[0]
    y = data[1]

    print(x.shape,y.shape)
    break

'''
DataLoader 默认用异步加载数据的方式来读取数据，一方面可以提升数据加载的速度，另一方面也会占据更少的内存。
如果你需要同时加载全部数据到内存中，请设置use_buffer_reader=False
'''