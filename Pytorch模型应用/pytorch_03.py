import torch
from torch import nn,optim

from torchvision import models

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()
        layer1 = nn.Sequential()
        layer1.add_module("conv1",nn.Conv2d(3,32,3 ,1,padding=1))
        layer1.add_module("relu1",nn.ReLU(True))
        layer1.add_module("pool1",nn.MaxPool2d(2,2))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module("conv2",nn.Conv2d(32,64,3,1,padding=1))
        layer2.add_module("relu2",nn.ReLU(True))
        layer2.add_module("pool2",nn.MaxPool2d(2,2))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module("conv3",nn.Conv2d(64,128,3,1,padding=1))
        layer3.add_module("relu3",nn.ReLU(True))
        layer3.add_module("pool3",nn.MaxPool2d(2,2))
        self.layer3 = layer3


        layer4 = nn.Sequential()
        layer4.add_module("fc1",nn.Linear(2048,512))
        layer4.add_module("fc1_relu",nn.ReLU(True))
        layer4.add_module("fc2",nn.Linear(512,64))
        layer4.add_module("fc2_relu",nn.ReLU(True))
        layer4.add_module("fc3",nn.Linear(64,10))
        self.layer4 = layer4


    def forward(self, x):
        conv1 = self.layer1(x)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        fc_input = conv3.view(conv3.size(0),-1)
        print(fc_input)
        conv4 = self.layer4(fc_input)

        return  conv4


if __name__=='__main__':

    input = torch.randn(3)
    model = SimpleCNN()
    # print(model.layer1.conv1)
    # print(model)
    print(*list(model.children())[:-1])
    # print(*(model.named_parameters()))
    # print(*list(model.named_modules()))
    # for m in model.modules():
    #     if isinstance(m,nn.Conv2d):
    #         # init.normal(m.weight.data )
    #         # initial parameters
    #         print("true")








# input = torch.randn(7,10,10)
#
# conv1 = nn.Conv1d(10,12,3,stride=2,padding=1)
#
# conv2 = nn.Conv2d(10,12,3,stride=2,padding=1)
#
# print(conv1(input))
# print(conv2)