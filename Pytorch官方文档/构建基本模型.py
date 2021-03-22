# -*- coding: utf-8 -*-
"""
Created on : 2021/3/16 16:40

@author: Jeremy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        # 等价于nn.Model.__init__(self)
        super(Net, self).__init__()

        # 输入1通道，输出6通道，卷积核5*5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 定义卷积层：输入6张特征图，输出16张特征图，卷积核5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 定义全连接层：线性连接(y = Wx + b)，16*5*5个节点连接到120个节点上
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 定义全连接层：线性连接(y = Wx + b)，120个节点连接到84个节点上
        self.fc2 = nn.Linear(120, 84)
        # 定义全连接层：线性连接(y = Wx + b)，84个节点连接到10个节点上
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 输入x->conv1->relu->2x2窗口的最大池化->更新到x
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 输入x->conv2->relu->2x2窗口的最大池化->更新到x
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # view函数将张量x变形成一维向量形式，总特征数不变，为全连接层做准备
        x = x.view(x.size()[0], -1)
        # 输入x->fc1->relu，更新到x
        x = F.relu(self.fc1(x))
        # 输入x->fc2->relu，更新到x
        x = F.relu(self.fc2(x))
        # 输入x->fc3，更新到x
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    net = Net()
    print(net)
    # print(*net.parameters())

    for name, parameters in net.named_parameters():
        print(name, "：", parameters.size())

    # #########网络传播过程模拟#########
    # 输入如果没有batch数，则少一维度，Tensor,unsqueeze()可以为张量扩维
    input_ = Variable(torch.randn(1, 1, 32, 32))
    out = net(input_)
    print(out.size())
    """
    torch.Size([1, 10])
    """
    # net.zero_grad()
    # 输出值为10个标量（一个向量），所以需要指定每个标量梯度的权重
    # out.backward(t.ones(1,10))

    # #########Loss设计#########
    target = torch.arange(0, 10)
    target = target.to(torch.float32)
    # Loss需要先实例化，然后是callable的实例
    loss_fn = nn.MSELoss()  # 均方误差
    loss = loss_fn(out, target)
    print(loss)

    net.zero_grad()
    print("反向传播之前：", net.conv1.bias.grad)
    loss.backward()
    print("反向传播之后：", net.conv1.bias.grad)

    # #########优化器设计#########
    print(net.parameters())
    """
    <generator object Module.parameters at 0x0000021B525BE888>
    """
    # 初始化优化器
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    optimizer.zero_grad()  # 效果等同net.zero_grad()

    output = net(input_)
    output = output.to(torch.float32)

    loss = loss_fn(output, target)

    loss.backward()
    print("反向传播之前：", net.conv1.bias.data)
    optimizer.step()
    print("反向传播之后：", net.conv1.bias.data)