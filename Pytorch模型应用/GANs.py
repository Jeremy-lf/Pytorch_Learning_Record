import torch
from torch import nn
from torch.autograd import Variable

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dis(x)
        return x

class generator(nn.Module):
    def __init__(self,input_size):
        super(generator,self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(input_size,256),
            nn.ReLU(True),
            nn.Linear(256,256),
            nn.ReLU(True),
            nn.Linear(256,784),
            nn.Tanh()  # -1~1
        )

    def forward(self,x):
        x = self.gen(x)
        return x

criterion = nn.BCELoss()
target = torch.empty(3).random_(2)

D = discriminator()
G = generator()
d_optimizer = torch.optim.Adam(D.parameters(),lr=0.0001)
g_optimizer = torch.optim.Adam(G.parameters(),lr=0.0001)


#discrimination training

img  = img.view(num_img,-1)
real_img = Variable(img)
real_label = torch.ones(num_img)
fake_label = torch.zeros(num_img)

# discrimination_model  training
real_out = D(real_img)
d_loss_real = criterion(real_out,real_label)
real_scores = real_out

z = torch.randn(num_img,z_dimension)
fake_img = G(z)
fake_out = D(fake_img)
d_loss_fake = criterion(fake_out,fake_label)
fake_scores = fake_out

d_loss = d_loss_real+d_loss_fake
d_optimizer.zero_grad()
d_loss.backward()
d_optimizer.step()


#computer the loss of fake_img  generator_model training
z = torch.randn(num_img,z_dimension)
fake_img = G(z)
output = D(fake_img)
g_loss = criterion(output,real_label)
g_optimizer.zero_grad()
g_loss.backward()
g_optimizer.step()

# more complicated network
class discriminator_1(nn.Module):
    def __init__(self):
        super(discriminator_1,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,32,5,padding=2), #batch 32 28 28
            nn.LeakyReLU(0.2),#if a<0  a =a*0.1
            nn.AvgPool2d(2,stride=2) #batch 32 14 14
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,5,padding=2),#batch 64 14 14
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2,stride=2)#batch 64 7 7
        )

        self.fc = nn.Sequential(
            nn.Linear(64*7*7,1024),
            nn.LeakyReLU(0.2,True),
            nn.Linear(1024,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        # x : batch w h c
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x

class generator_1(nn.Module):
    def __init__(self,input_size,num_feature):
        super(generator_1,self).__init__()
        self.fc = nn.Linear(input_size,num_feature)# batch 31365 = 1*56*56
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )

        self.downsample1 = nn.Sequential(
            nn.Conv2d(1,50,3,stride=1,padding=1),#batch 50 56 56
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )

        self.downsample2 = nn.Sequential(
            nn.Conv2d(50,25,3,stride=1,padding=1),#batch 25 56 56
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )

        self.downsample3 = nn.Sequential(
            nn.Conv2d(25,1,3,stride=2),#batch 1 28 28
            nn.Tanh()
        )

    def forward(self,x):
        x = self.fc(x)
        x = x.view(x.size(0),1,56,56)
        x = self.br(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)

        return x


