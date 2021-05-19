# -*- coding: utf-8 -*-
"""
Created on : 2021/1/18 16:19

@author: Jeremy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# 定义模型
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleNet(nn.Module):
    def __init__(self,in_put,hindden,out_put):
        super(SimpleNet,self).__init__()
        self.layers1 = nn.Linear(in_put,hindden)

        self.BatchNormal = nn.BatchNorm1d(hindden)
        self.Relu = nn.ReLU(hindden)  # method one
        self.layers2 = nn.Linear(hindden,hindden)
        self.layers3 = nn.Linear(hindden,out_put)

        self.layers4 = nn.Sequential(
            nn.Linear(in_put,hindden),
            nn.BatchNorm1d(hindden),
            nn.ReLU(True),  #method two
            nn.Linear(hindden,hindden),
            nn.Linear(hindden,out_put)
        )
    def forward(self, x):
        x = self.layers1(x)
        x = self.Relu(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)
        return x
# 初始化模型
model = TheModelClass()
model2 = SimpleNet(2,4,2)
# 初始化优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 打印模型的状态字典
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# 打印优化器的状态字典
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

'''
load_state_dict()函数只接受字典对象，而不是保存对象的路径。
这就意味着在你传给load_state_dict()函数之前，你必须反序列化 你保存的state_dict。
例如，你无法通过 model.load_state_dict(PATH)来加载模型。
'''

# dictionary = {0:'我',1:'爱','2':'你'}
# save_file = {"model":model.state_dict(),"dict": dictionary}



# torch.save(save_file,'setmodel.pth')
# torch.save(model,'TheNet.pth')
# torch.save(model2,'SimpleNet.pth')

# checkpoint = torch.load('setmodel.pth')
#
# model_dict = checkpoint['model']
# print(model.load_state_dict(model_dict))
# print(checkpoint["dict"])

print(*(model.parameters()))

# torch.save({"encoder":model,"decoder":model2},"encoder_decoder_1.pth")
# torch.save({"encoder":model.state_dict(),"decoder":model2.state_dict()},"encoder_decoder_1.pth")
#
#
# checkpoint = torch.load('encoder_decoder_1.pth')
#
# # 加载模型的两种方式
# # encodermodel = model.load_state_dict((checkpoint["encoder"]).state_dict())
# encodermodel = model.load_state_dict(checkpoint["encoder"])
# print(encodermodel)