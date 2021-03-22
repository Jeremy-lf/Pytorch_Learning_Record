import torch
from torch import nn

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28,128),
            nn.ReLU(True),
            nn.Linear(128,64),
            nn.ReLU(True),
            nn.Linear(64,12),
            nn.ReLU(True),
            nn.Linear(12,3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3,12),
            nn.ReLU(True),
            nn.Linear(12,64),
            nn.ReLU(64,128),
            nn.Linear(128,28*28),
            nn.Tanh()
        )

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class DCautoencoder(nn.Module):
    def __init__(self):
        super(DCautoencoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,3,stride=3,padding=1), # b 16 10 10
            nn.ReLU(True),
            nn.MaxPool2d(2,stride=2),# b 16 5 5
            nn.Conv2d(16,8,3,stride=2,padding=1), # b 8 3 3
            nn.ReLU(True),
            nn.MaxPool2d(2,stride=1)# b 8 2 2
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8,16,3,stride=2),# b 16 5 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16,8,5,stride=3,padding=1), #b 8 15 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8,1,2,stride=2,padding=1),#b 1 28 28
            nn.Tanh()
        )

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x



