# -*- coding: utf-8 -*-
"""
Created on : 2021/5/19 16:46

@author: Jeremy
"""
import paddle
print('数据处理方法：', paddle.vision.transforms.__all__)

# 数据处理方法： ['BaseTransform', 'Compose', 'Resize', 'RandomResizedCrop', 'CenterCrop', 'RandomHorizontalFlip',
# 'RandomVerticalFlip', 'Transpose', 'Normalize', 'BrightnessTransform', 'SaturationTransform', 'ContrastTransform',
# 'HueTransform', 'ColorJitter', 'RandomCrop', 'Pad', 'RandomRotation', 'Grayscale', 'ToTensor', 'to_tensor',
# 'hflip', 'vflip', 'resize', 'pad', 'rotate', 'to_grayscale', 'crop', 'center_crop', 'adjust_brightness',
# 'adjust_contrast', 'adjust_hue', 'normalize']

from paddle.vision.transforms import Compose,Resize,ColorJitter
# 定义想要使用的数据增强方式，这里包括随机调整亮度、对比度和饱和度，改变图片大小
transform = Compose([ColorJitter(),Resize(size=32)])

import paddle
from paddle.io import Dataset
from paddle.vision.transforms import Compose,Resize
import cv2


BATCH_SIZE = 64
BATCH_NUM = 20

IMAGE_SIZE = (28, 28)
CLASS_NUM = 10

class MyDataset(Dataset):
    def __init__(self, num_samples):
        super(MyDataset, self).__init__()
        self.num_samples = num_samples
        # 在 `__init__` 中定义数据增强方法，此处为调整图像大小
        self.transform = Compose([Resize(size=32)])

    def __getitem__(self, index):
        data = paddle.uniform(IMAGE_SIZE,dtype='float32')
        # 在 `__getitem__` 中对数据集使用数据增强方法
        data = self.transform(data.numpy())
        label = paddle.randint(0,CLASS_NUM-1,dtype='int64')

        return data,label

    def __len__(self):
        return self.num_samples


