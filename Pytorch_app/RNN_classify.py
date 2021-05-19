import torch
from torch import nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self,in_dim,hidden_dim,n_layer,n_class):
        super(RNN,self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim,hidden_dim,n_layer,batch_first=True)
        self.calssifier = nn.Linear(hidden_dim,n_class)

    def forward(self, x):
        # h_0 = Variable(torch.zeros(self.n_layer,x.size(1),self.hidden_dim)) #.cuda()
        # c_0 = Variable(torch.zeros(self.n_layer,x.size(1),self.hidden_dim)) #.cuda()
        out,_ = self.lstm(x)
        out = out[:,-1,:]
        out = self.calssifier(out)
        return out



class LSTM(nn.Module):
    def __init__(self,input_size=2,hidden_size=4,output_size=1,num_layer=2):
        super(LSTM,self).__init__()

        self.layer1 = nn.LSTM(input_size,hidden_size,num_layer)
        self.layer2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x,_ = self.layer1(x) #seq batch hidden
        s,b,h = x.size()
        x = x.view(s*b,h)
        x = self.layer2(x)
        x = x.view(s,b,-1)
        return x



if __name__=="__main__":
    model=RNN(32,4,1,3)
    x = torch.randn(10,32,32)  #(batch seq input.dim)
    print(model)
    print(model(x))
    print(model(x).shape)