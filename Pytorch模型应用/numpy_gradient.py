# -*- coding: utf-8 -*-
"""
Created on : 2021/1/18 21:46

@author: Jeremy
"""
import numpy as np
# N是批量大小; D_in是输入维度;
# 49/5000 H是隐藏的维度; D_out是输出维度。
N, D_in, H, D_out = 64, 1000, 100, 10
# 创建随机输入和输出数据
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)
# 随机初始化权重
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)
learning_rate = 1e-6
for t in range(500):
    # 前向传递：计算预测值y
    h = x.dot(w1)
    h_relu = np.maximum(h,0)
    y_pred = h_relu.dot(w2)

    # 计算和打印损失loss
    # loss = np.square(y_pred - y).sum().item()
    loss = np.sum(np.square(y_pred - y))

    # print(t, loss)
    # 反向传播，计算w1和w2对loss的梯度
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)
    # 更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

    print("t",t,"y_pred",y_pred)


import torch
# N是批大小；D是输入维度
# H是隐藏层维度；D_out是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10
#创建输入和输出随机张量
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
# 使用nn包将我们的模型定义为一系列的层。
# nn.Sequential是包含其他模块的模块，并按顺序应用这些模块来产生其输出。
# 每个线性模块使用线性函数从输入计算输出，并保存其内部的权重和偏差张量。
# 在构造模型之后，我们使用.to()方法将其移动到所需的设备。
model = torch.nn.Sequential(
torch.nn.Linear(D_in, H),
torch.nn.ReLU(),
torch.nn.Linear(H, D_out),
)
# nn包还包含常用的损失函数的定义；
# 在这种情况下，我们将使用平均平方误差(MSE)作为我们的损失函数。
# 设置reduction='sum'，表示我们计算的是平方误差的“和”，而不是平均值;
# 这是为了与前面我们手工计算损失的例子保持一致，
# 但是在实践中，通过设置reduction='elementwise_mean'来使用均方误差作为损失更为常见。
loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-4
for t in range(500):
    # 前向传播：通过向模型传入x计算预测的y。
    # 模块对象重载了__call__运算符，所以可以像函数那样调用它们。
    # 这么做相当于向模块传入了一个张量，然后它返回了一个输出张量。
    y_pred = model(x)
    # 计算并打印损失。
    # 传递包含y的预测值和真实值的张量，损失函数返回包含损失的张量。
    loss = loss_fn(y_pred, y)
    print(t, loss.item())
    # 反向传播之前清零梯度
    model.zero_grad()
    # 反向传播：计算模型的损失对所有可学习参数的导数（梯度）。
    # 在内部，每个模块的参数存储在requires_grad=True的张量中，
    # 因此这个调用将计算模型中所有可学习参数的梯度。
    loss.backward()
    # 使用梯度下降更新权重。
    # 每个参数都是张量，所以我们可以像我们以前那样可以得到它的数值和梯度
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad