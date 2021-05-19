import torch
from torch import optim,nn
from torch.autograd import Variable
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

import pytorch_01


#Hyperparameters

batch_size = 16
learnning_rate = 0.001
num_epochs = 20

data_tf = transforms.Compose([transforms.ToTensor(),transforms.Normalize( [0.5],  [0.5J])])

train_dataset  =  datasets.MNIST(root='./data', train=True,transform=data_tf,download=True)
test_dataset  =  datasets.MNIST(root='./data', train=False,transform=data_tf)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#model
model = pytorch_01.SimpleNet_1(28*28,100,10)

if torch.cuda.is_available():
    model = model.cuda()

#loss
criterion  =  nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(),lr=learnning_rate)

#train
train_loss = 0
num_correct = 0
train_acc =0
for num in range(num_epochs):
    model = model.train()
    for data in train_loader:
        img,label = data
        img = img.view(img.size(0),-1)

        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()

        out = model(img)

        loss= criterion(out,label)

        train_loss += loss.data[0]*label.size(0)
        _,pred = torch.max(out,1)
        num_correct = (pred==label).sum()
        train_acc += num_correct.data[0]

    print("Epoch:%d,loss:%.2f,acc:%4f" % (num,train_loss/len(train_loader),train_acc/len(train_loader)))





