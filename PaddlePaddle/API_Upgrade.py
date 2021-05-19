# -*- coding: utf-8 -*-
"""
Created on : 2021/5/18 21:15

@author: Jeremy
"""


'''
增加了paddle.Model高层API，大部分任务可以使用此API用于简化训练、评估、预测类代码开发。
注意区别Model和Net概念，Net是指继承paddle.nn.Layer的网络结构；
而Model是指持有一个Net对象，同时指定损失函数、优化算法、评估指标的可训练、评估、预测的实例。
'''

import paddle

from paddle.vision.transforms import transforms
from paddle.io import DataLoader
#
transform = transforms.Compose([transforms.ToTensor()])
#
# train_data = paddle.vision.datasets.MNIST(mode='train',transform = transform)
# test_data = paddle.vision.datasets.MNIST(mode='test',transform= transform)
# lenet = paddle.vision.models.LeNet()
#
# # Mnist继承paddle.nn.Layer属于Net，model包含了训练功能
#
# model = paddle.Model(lenet)
# # 设置训练模型所需的optimizer, loss, metric
# model.prepare(
#     paddle.optimizer.Adam(learning_rate=0.001,parameters=model.parameters()),
#     paddle.nn.CrossEntropyLoss(),
#     paddle.metric.Accuracy()
# )
#
# # 启动训练
# model.fit(train_data,epochs=2,batch_size=32,log_freq=200)
#
# # 启动评估
# model.evaluate(test_data,log_freq=20,batch_size=32)

'''
使用基础API
'''

import paddle
from paddle.vision.transforms import ToTensor

train_data = paddle.vision.datasets.MNIST(mode='train',transform = transform)
test_data = paddle.vision.datasets.MNIST(mode='test',transform= transform)
lenet = paddle.vision.models.LeNet()


materion = paddle.nn.CrossEntropyLoss()

train_loader = DataLoader(train_data,batch_size=32,shuffle=True)
optim = paddle.optimizer.Adam(learning_rate=0.001,parameters=lenet.parameters())
epochs = 2


def train():
    for epoch in range(epochs):
        for i,data in enumerate(train_loader):
            x_data = data[0]
            y_data = data[1]

            output = lenet(x_data)
            acc = paddle.metric.accuracy(output,y_data)
            loss = materion(output,y_data)
            loss.backward()
            if i % 100==0:
                print("epoch:{},batch:{},loss is：{},acc is:{}" .format(epoch,i,loss.numpy(),acc.numpy()))
            optim.step()
            optim.clear_grad()
train()


'''
单机多卡启动
'''

'''
当调用paddle.Model高层来实现训练时，想要启动单机多卡训练非常简单，代码不需要做任何修改，
只需要在启动时增加一下参数-m paddle.distributed.launch。
'''

'''
# 单机单卡启动，默认使用第0号卡
$ python train.py

# 单机多卡启动，默认使用当前可见的所有卡
$ python -m paddle.distributed.launch train.py

# 单机多卡启动，设置当前使用的第0号和第1号卡
$ python -m paddle.distributed.launch --selected_gpus='0,1' train.py

# 单机多卡启动，设置当前使用第0号和第1号卡
$ export CUDA_VISIBLE_DEVICES=0,1
$ python -m paddle.distributed.launch train.py
'''


import paddle
from paddle.vision.transforms import ToTensor

# 第1处改动，导入分布式训练所需要的包
import paddle.distributed as dist

train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=ToTensor())
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=ToTensor())
lenet = paddle.vision.models.LeNet()
loss_fn = paddle.nn.CrossEntropyLoss()

# 加载训练集 batch_size 设为 64
train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)

def train(model):
    # 第2处改动，初始化并行环境
    dist.init_parallel_env()

    # 第3处改动，增加paddle.DataParallel封装
    lenet = paddle.DataParallel(model)
    epochs = 2
    adam = paddle.optimizer.Adam(learning_rate=0.001, parameters=lenet.parameters())
    # 用Adam作为优化函数
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader):
            x_data = data[0]
            y_data = data[1]
            predicts = lenet(x_data)
            acc = paddle.metric.accuracy(predicts, y_data)
            loss = loss_fn(predicts, y_data)
            loss.backward()
            if batch_id % 100 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
            adam.step()
            adam.clear_grad()

# 启动训练
train(lenet)
