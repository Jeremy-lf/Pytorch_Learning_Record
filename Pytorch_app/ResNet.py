from torch import nn
import torch


def conv3x3(in_plans,out_plan,stride=1):
    #3x3  convolution with padding
    return nn.Conv2d(in_plans,out_plan,kernel_size=3,stride=stride,padding=1,bias=False)



class BasicBlock(nn.Module):
    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(BasicBlock,self).__init__()
        self.conv1 = conv3x3(inplanes,planes,stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(True)

        self.conv2 = conv3x3(planes,planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out +=residual
        print(out)
        out = self.relu1(out)

        return out


class ResNet(nn.Module):
    def __init__(self,block,layers,num_class=10):
        super(ResNet,self).__init__()
        self.in_channel =16
        self.conv = conv3x3(3,16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(True)

        self.layer1 = self.make_layer(block,16,layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1],2)
        self.layer3 = self.make_layer(block, 64, layers[2],2)

        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64,num_class)



    def make_layer(self,block,out_channels,blocks,stride=1):
        downsample = None
        if (stride!=1) or (self.in_channel!=out_channels):
            downsample = nn.Sequential(conv3x3(self.in_channel,out_channels,stride=stride),
                                       nn.BatchNorm2d(out_channels))

        layers =[]
        layers.append(
            block(self.in_channel,out_channels,stride,downsample)
        )
        self.in_channel = out_channels

        for i in range(1,blocks):
            layers.append(block(out_channels,out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out


if __name__=="__main__":
    Block = [1,2,3]
    print(*Block)
    model = ResNet(BasicBlock,[2,2,3])
    print(model)
