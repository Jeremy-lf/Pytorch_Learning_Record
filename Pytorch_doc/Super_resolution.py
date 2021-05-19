# -*- coding: utf-8 -*-
"""
Created on : 2021/3/21 19:35

@author: Jeremy
"""
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils import model_zoo
import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend
import numpy as np

class SuperResolutionNet(nn.Module):

    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))

        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor) # 上采样
        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

# 使用上面模型定义，创建super-resolution模型
torch_model = SuperResolutionNet(upscale_factor=3)


# 加载预先训练好的模型权重
# model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
batch_size = 1 # just a random number
# 使用预训练的权重初始化模型
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
    # torch_model.load_state_dict(model_zoo.load_url(model_url,map_location=map_location))
    torch_model.load_state_dict(torch.load('./superres_epoch100-44c6958e.pth'))
# 将训练模式设置为falsesince we will only run the forward pass.
torch_model.train(False)


# 输入模型
x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
# 导出模型
torch_out = torch.onnx._export(torch_model,x,"super_resolution.onnx",export_params=True,keep_initializers_as_inputs=True)
# store the trained parameter weights inside the model file

#加载ONNX ModelProto对象。模型是一个标准的Python protobuf对象
model = onnx.load("super_resolution.onnx")
# 为执行模型准备caffe2后端，将ONNX模型转换为可以执行它的Caffe2 NetDef。
# 其他ONNX后端，如CNTK的后端即将推出。
prepared_backend = onnx_caffe2_backend.prepare(model)
# 在Caffe2中运行模型
# 构造从输入名称到Tensor数据的映射。
# 模型图形本身包含输入图像之后所有权重参数的输入。由于权重已经嵌入，我们只需要传递输入图像。

# 设置第一个输入。
W = {model.graph.input[0].name: x.data.numpy()}
# 运行Caffe2 net:
c2_out = prepared_backend.run(W)[0]
# 验证数字正确性，最多3位小数
np.testing.assert_almost_equal(torch_out.data.cpu().numpy(), c2_out, decimal=3)
print("Exported model has been executed on Caffe2 backend, and the result looksgood!")

# 从内部表示中提取工作空间和模型原型
c2_workspace = prepared_backend.workspace
c2_model = prepared_backend.predict_net
# 现在导入caffe2的`mobile_exporter`
from caffe2.python.predictor import mobile_exporter
# 调用Export来获取predict_net，init_net。 在移动设备上运行时需要这些网络
init_net, predict_net = mobile_exporter.Export(c2_workspace, c2_model,c2_model.external_input)
# 我们还将init_net和predict_net保存到我们稍后将用于在移动设备上运行它们的文件中
with open('init_net.pb', "wb") as fopen:
    fopen.write(init_net.SerializeToString())
with open('predict_net.pb', "wb") as fopen:
    fopen.write(predict_net.SerializeToString())


# 一些必备的导入包
from caffe2.proto import caffe2_pb2
from caffe2.python import core, net_drawer, net_printer, visualize, workspace,utils

import numpy as np
import os
import subprocess
from PIL import Image
from matplotlib import pyplot
from skimage import io, transform

# 加载图像
img_in = io.imread("./_static/img/cat.jpg")
# 设置图片分辨率为 224x224
img = transform.resize(img_in, [224, 224])
# 保存好设置的图片作为模型的输入
io.imsave("./_static/img/cat_224x224.jpg", img)

# 加载设置好的图片并更改为YCbCr的格式
img = Image.open("./_static/img/cat_224x224.jpg")
img_ycbcr = img.convert('YCbCr')
img_y, img_cb, img_cr = img_ycbcr.split()
# 让我们运行上面生成的移动网络，以便正确初始化caffe2工作区
workspace.RunNetOnce(init_net)
workspace.RunNetOnce(predict_net)
# Caffe2有一个很好的net_printer能够检查网络的外观
# 并确定我们的输入和输出blob名称是什么。
print(net_printer.to_string(predict_net))

# 现在，让我们传递调整大小的猫图像以供模型处理。
workspace.FeedBlob("9", np.array(img_y)[np.newaxis,
np.newaxis, :, :].astype(np.float32))
# 运行predict_net以获取模型输出
workspace.RunNetOnce(predict_net)
# 现在让我们得到模型输出blob
img_out = workspace.FetchBlob("27")

img_out_y = Image.fromarray(np.uint8((img_out[0, 0]).clip(0, 255)), mode='L')
# 获取输出图像遵循PyTorch实现的后处理步骤
final_img = Image.merge("YCbCr", [img_out_y,
img_cb.resize(img_out_y.size, Image.BICUBIC),
img_cr.resize(img_out_y.size, Image.BICUBIC),
]).convert("RGB")
# 保存图像，我们将其与移动设备的输出图像进行比较
final_img.save("./_static/img/cat_superres.jpg")
