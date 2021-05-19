from torch import nn
import torch
from torchvision import models
from torch.autograd import Variable
from torch import optim

transfer_model = models.resnet18(pretrained=False)

# first
dim_in = transfer_model.fc.in_features
transfer_model.fc = nn.Linear(dim_in,10) #img_class =10
# print(transfer_model)

# second
for param in transfer_model.parameters():
    param.requires_grad = False
optimizer = optim.SGD(transfer_model.fc.parameters(),lr=1e-3)

#third
class feature_net(nn.Module):
    def __init__(self,model):
        super(feature_net,self).__init__()

        if model =='vgg':
            vgg = models.vgg19()
            self.feature = nn.Sequential(*list(vgg.children())[:-1])
            self.feature.add_module('global average',nn.AvgPool2d(9))
        elif model =='inceptionv3':
            inception = models.inception_v3()
            self.feature = nn.Sequential(*list(inception.children())[:-1])
            self.feature._modules.pop('13')
            self.feature.add_module('global average',nn.AvgPool2d(35))
        elif model =='resnet152':
            resnet = models.resnet152()
            self.feature = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self,x):
        x = self.feature(x)
        x = x.view(x.size(0),-1)
        return x

class classifier(nn.Module):
    def __init__(self,dim,num_class):
        super(classifier,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim,1000),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1000,num_class)
        )

    def forward(self, x):
        x = self.fc(x)
        return x




if __name__=='__main__':
    # model = feature_net(model='vgg')
    # print(model)

    model_1 = models.vgg19(pretrained=False)
    print(model_1)

    model = models.resnet18().fc.in_features