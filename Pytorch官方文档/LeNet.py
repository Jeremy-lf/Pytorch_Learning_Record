# -*- coding: utf-8 -*-
"""
Created on : 2021/3/17 12:22

@author: Jeremy
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F

'''
初始化网络
初始化Loss函数 & 优化器
进入step循环：
　　梯度清零
　　向前传播
　　计算本次Loss
　　向后传播
　　更新参数
'''
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    net = LeNet()

    # #########训练网络#########
    from torch import optim

    # 初始化Loss函数 & 优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        running_loss = 0.0
        for step, data in enumerate(trainloader, 0):  # step为训练次数， trainloader包含batch的数据和标签
            inputs, labels = data
            inputs, labels = t.autograd.Variable(inputs), t.autograd.Variable(labels)

            # 梯度清零
            optimizer.zero_grad()

            # forward
            outputs = net(inputs)
            # backward
            loss = loss_fn(outputs, labels)
            loss.backward()
            # update
            optimizer.step()

            running_loss += loss.data[0]
            if step % 2000 == 1999:
                print("[{0:d}, {1:5d}] loss: {2:3f}".format(epoch + 1, step + 1, running_loss / 2000))
                running_loss = 0.
    print("Finished Training")