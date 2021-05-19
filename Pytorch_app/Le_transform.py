# -*- coding: utf-8 -*-
"""
Created on : 2021/4/29 14:45

@author: Jeremy
"""
from PIL import Image
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from torch.autograd import Variable
from torchvision.transforms import functional as F

# python的用法->tensor数据类型
# 通过transforms.ToTensor去看两个问题

# 绝对路径：D:\leran_pytorch\dataset\train\ants\0013035.jpg
# 相对路径：dataset/train/ants/0013035.jpg

img_path = "./k.jpg"
img = Image.open(img_path)

# writer = SummaryWriter("logs")

# 1、transforms该如何使用（python）
# 2、为什么我们需要Tensor数据类型
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

tensor_img1 = F.to_tensor(img)  #等价于ToTensor()

print(tensor_img.type(),tensor_img1.type())
print(tensor_img.shape)

'''
transforms.Normalize使用如下公式进行归一化：
channel=（channel-mean）/std(因为transforms.ToTensor()已经把数据处理成[0,1],那么(x-0.5)/0.5就是[-1.0, 1.0])
'''

# writer.add_image("Tensor_img", tensor_img)
# writer.close()


# import matplotlib.pyplot as plt

