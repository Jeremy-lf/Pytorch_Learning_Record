# -*- coding: utf-8 -*-
"""
Created on : 2021/5/19 13:44

@author: Jeremy
"""
from paddle.static import InputSpec


'''
Paddle保存的模型有两种格式，一种是训练格式，保存模型参数和优化器相关的状态，可用于恢复训练；
一种是预测格式，保存预测的静态图网络结构以及参数，用于预测部署。
'''

'''
高层API场景
高层API下用于预测部署的模型保存方法为：

model = paddle.Model(Mnist())
# 预测格式，保存的模型可用于预测部署
model.save('mnist', training=False)
# 保存后可以得到预测部署所需要的模型
'''

import paddle

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

# mnist = Mnist()
#
# input = InputSpec([-1, 1, 28, 28], 'float32', 'image')
# label = InputSpec([None, 1], 'int64', 'label')
#
# model = paddle.Model(Mnist(),input,label)
# # 预测格式，保存的模型可用于预测部署
# model.save('mnist', training=False)
# # 保存后可以得到预测部署所需要的模型


'''
基础API场景
'''

'''
动态图训练的模型，可以通过动静转换功能，转换为可部署的静态图模型，具体做法如下：
'''

import paddle
from paddle.jit import to_static
from paddle.static import InputSpec

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = paddle.nn.Linear(10, 3)

    # 第1处改动
    # 通过InputSpec指定输入数据的形状，None表示可变长
    # 通过to_static装饰器将动态图转换为静态图Program
    @to_static(input_spec=[InputSpec(shape=[None, 10], name='x'), InputSpec(shape=[3], name='y')])
    def forward(self, x, y):
        out = self.linear(x)
        out = out + y
        return out


net = SimpleNet()

# 第2处改动
# 保存静态图模型，可用于预测部署
paddle.jit.save(net, './simple_net')