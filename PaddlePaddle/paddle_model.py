# -*- coding: utf-8 -*-
"""
Created on : 2021/4/28 20:03

@author: Jeremy
"""
import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
from paddle.static import InputSpec
"""


device = paddle.set_device('cpu')  # or 'gpu'

net = nn.Sequential(
    nn.Flatten(1),
    nn.Linear(784, 200),
    nn.Tanh(),
    nn.Linear(200, 10))

# inputs and labels are not required for dynamic graph.
input = InputSpec([None, 784], 'float32', 'x')
label = InputSpec([None, 1], 'int64', 'label')

model = paddle.Model(net, input, label)
optim = paddle.optimizer.SGD(learning_rate=1e-3,
                             parameters=model.parameters())
model.prepare(optim,
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy())

transform = T.Compose([
    T.Transpose(),
    T.Normalize([127.5], [127.5])
])
data = paddle.vision.datasets.MNIST(mode='train', transform=transform)
model.fit(data, epochs=2, batch_size=32, verbose=1)

"""

import numpy as np
import paddle
from paddle.static import InputSpec

class MnistDataset(paddle.vision.datasets.MNIST):
    def __init__(self, mode, return_label=True):
        super(MnistDataset, self).__init__(mode=mode)
        self.return_label = return_label

    def __getitem__(self, idx):
        img = np.reshape(self.images[idx], [1, 28, 28])
        if self.return_label:
            return img, np.array(self.labels[idx]).astype('int64')
        return img,

    def __len__(self):
        return len(self.images)

# test_dataset = MnistDataset(mode='test', return_label=False)
#
# # imperative mode
# input = InputSpec([-1, 1, 28, 28], 'float32', 'image')
# model = paddle.Model(paddle.vision.models.LeNet(), input)
# model.prepare()
# result = model.predict(test_dataset, batch_size=64)
# print(len(result[0]), result[0][0].shape)
#
# # declarative mode
# device = paddle.set_device('cpu')
# paddle.enable_static()
# input = InputSpec([-1, 1, 28, 28], 'float32', 'image')
# model = paddle.Model(paddle.vision.models.LeNet(), input)
# model.prepare()
#
# result = model.predict(test_dataset, batch_size=64)
# print(len(result[0]), result[0][0].shape)

# from paddle.vision.datasets import MNIST
#
# mnist = MNIST(mode='test')
#
# for i in range(len(mnist)):
#     sample = mnist[i]
#     print(sample[0].size, sample[1])

import paddle
import paddle.vision.transforms as transform
from paddle.static import InputSpec

# declarative mode
transforms = transform.Compose([
    T.Transpose(),
    T.Normalize([127.5], [127.5])
])
val_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transforms)

input = InputSpec([-1, 1, 28, 28], 'float32', 'image')
label = InputSpec([None, 1], 'int64', 'label')
model = paddle.Model(paddle.vision.models.LeNet(), input, label)
model.prepare(metrics=paddle.metric.Accuracy())
result = model.evaluate(val_dataset, batch_size=64)
print(result)