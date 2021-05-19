# -*- coding: utf-8 -*-
"""
Created on : 2021/3/17 12:22

@author: Jeremy
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F

import torch
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
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""
#
#     def __call__(self, sample):
#         image, landmarks = sample['image'], sample['landmarks']
#
#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C X H X W
#         image = image.transpose((2, 0, 1))
#         return {'image': torch.from_numpy(image),
#                 'landmarks': torch.from_numpy(landmarks)}

if __name__ == "__main__":
    net = LeNet()

    # #########训练网络#########
    from torch import optim
    # from torchvision.datasets import MNIST
    import  torchvision
    import numpy
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from torch.autograd import Variable

    # 初始化Loss函数 & 优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # transforms = transforms.Compose([])

    DOWNLOAD = False
    BATCH_SIZE = 32
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化

    train_dataset = torchvision.datasets.MNIST(root='./', train=True, transform=transform, download=DOWNLOAD)
    test_dataset = torchvision.datasets.MNIST(root='./data/mnist',
                                              train=False,
                                              transform=torchvision.transforms.ToTensor(),
                                              download=True)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

    for epoch in range(200):
        running_loss = 0.0
        for step, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = t.autograd.Variable(inputs), t.autograd.Variable(labels)
            # inputs = torch.from_numpy(inputs).unsqueeze(1)
            # labels = torch.from_numpy(numpy.array(labels))
            # 梯度清零
            optimizer.zero_grad()

            # forward
            outputs = net(inputs)
            # backward
            loss = loss_fn(outputs, labels)
            loss.backward()
            # update
            optimizer.step()

            running_loss += loss.item()
            if step % 10 == 9:
                print("[{0:d}, {1:5d}] loss: {2:3f}".format(epoch + 1, step + 1, running_loss / 2000))
                running_loss = 0.
    print("Finished Training")

    # save the trained net
    torch.save(net, 'net.pkl')

    # load the trained net
    net1 = torch.load('net.pkl')

    # test the trained net
    correct = 0
    total = 1
    for images, labels in test_loader:
        preds = net(images)
        predicted = torch.argmax(preds, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print('accuracy of test data:{:.1%}'.format(accuracy))



