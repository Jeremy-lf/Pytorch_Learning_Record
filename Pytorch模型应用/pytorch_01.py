import cv2

import torch.nn as nn


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



class SimpleNet_1(nn.Module):
    def __init__(self,input,hindden,output):
        super(SimpleNet_1, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input,hindden),nn.BatchNorm1d(hindden),nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(hindden,hindden),nn.BatchNorm1d(hindden),nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(hindden,output))

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x



if __name__=="__main__":
    # net = SimpleNet(in_put=1,hindden=20,out_put=5)
    net = SimpleNet(1,20,4)
    net_1 = SimpleNet_1(1,20,4)
    # print(net.parameters())
    # print(net_1.parameters())
    # cv2.imread()
    print(net)
    print(net.__getattr__('layers1'))
    # print(net_1.state_dict())

    for k,v in enumerate(net_1.state_dict()):
        print(k,v)