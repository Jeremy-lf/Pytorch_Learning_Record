#GoogleNet

from torch import nn
import torch

class BasicConv2d(nn.Module):
    def __init__(self,in_channel,out_channnel,**kwargs):
        super(BasicConv2d,self).__init__()
        self.conv = nn.Conv2d(in_channel,out_channnel,bias=False,**kwargs)
        self.bn = nn.BatchNorm2d(out_channnel,eps=0.001)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = nn.ReLU(True)

        return x

class Inception(nn.Module):
    def __init__(self,in_channel,pool_feature):
        super(Inception,self).__init__()

        self.branch1x1 = BasicConv2d(in_channel,64,kernel_size =1)

        self.branch5x5_1 = BasicConv2d(in_channel,48,kernel_size =1)
        self.branch5x5_2 = BasicConv2d(48,64,kernel_size =5,padding=2)

        self.branch3x3db1_1 = BasicConv2d(in_channel,64,kernel_size=1)
        self.branch3x3db1_2 = BasicConv2d(64,96,kernel_size=3,padding=1)
        self.branch3x3db1_3 = BasicConv2d(96,96,kernel_size=3,padding=1)

        self.branch_pool = BasicConv2d(in_channel,pool_feature,kernel_size=1)

        self.AvgPool2d = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3db1 = self.branch3x3db1_1(x)
        branch3x3db1 = self.branch3x3db1_2(branch3x3db1)
        branch3x3db1 = self.branch3x3db1_3(branch3x3db1)

        branch_pool = self.AvgPool2d(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1,branch5x5,branch3x3db1,branch_pool]

        return torch.cat(outputs,1)


if __name__=='__main__':

    model = Inception(10,100)
    print(model)
