# -*- coding: utf-8 -*-
"""
Created on : 2021/5/10 17:37

@author: Jeremy
"""
import paddle
from paddle.vision.transforms import Normalize
import cv2
import numpy as np

transform = Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')
# 使用transform对数据集做归一化
print('download training data and load training data')
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
print('load finished')

"""


train_data0, train_label_0 = train_dataset[0][0],train_dataset[0][1]
# train_data0 = train_data0.reshape([28,28])
train_data0 = train_data0.transpose(1,2,0)
cv2.imshow("m",train_data0)
cv2.waitKey(0)
"""

# 定义多层感知机
class MultilayerPerceptron(paddle.nn.Layer):
    def __init__(self,in_features):
        super(MultilayerPerceptron,self).__init__()

        # 形状变换
        self.flatten = paddle.nn.Flatten()
        # 第一个全连接层
        self.linear1 = paddle.nn.Linear(in_features=in_features, out_features=100)
        # 使用ReLU激活函数
        self.act1 = paddle.nn.ReLU()
        # 第二个全连接层
        self.linear2 = paddle.nn.Linear(in_features=100, out_features=100)
        # 使用ReLU激活函数
        self.act2 = paddle.nn.ReLU()
        # 第三个全连接层
        self.linear3 = paddle.nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        x = self.linear3(x)
        return x

# 使用 paddle.Model 封装 MultilayerPerceptron
net = MultilayerPerceptron(in_features=784)
print(net)
model = paddle.Model(net)
# 使用 summary 打印模型结构
# model.summary((-1, 1, 28, 28))

# 配置模型
model.prepare(paddle.optimizer.Adam(parameters=model.parameters()),  # 使用Adam算法进行优化
              paddle.nn.CrossEntropyLoss(), # 使用CrossEntropyLoss 计算损失
              paddle.metric.Accuracy()) # 使用Accuracy 计算精度


# 开始模型训练
model.fit(train_dataset, # 设置训练数据集
          epochs=5,      # 设置训练轮数
          batch_size=4, # 设置 batch_size
          verbose=1)     # 设置日志打印格式

model.evaluate(test_dataset, verbose=1)

results = model.predict(test_dataset)

# print(results)

# 获取概率最大的label
lab = np.argsort(results)                               #argsort函数返回的是result数组值从小到大的索引值
# print(lab)
print("该图片的预测结果的label为: %d" % lab[0][0][-1][0])  #-1代表读取数组中倒数第一列