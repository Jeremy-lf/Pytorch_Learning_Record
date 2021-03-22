import torch
from torch import nn
from torch.autograd import Variable


# RNN
basic_rnn = nn.RNN(input_size = 20,hidden_size=50,num_layers=2)

toy_input = Variable(torch.randn(100,32,20))  #seq batch feature.dim
h_0 = Variable(torch.randn(2,32,50))  #  layer*direction, batch, hinddem_size
toy_output,h_n = basic_rnn(toy_input)

print("RNN out:")
print(toy_output.size()) #torch.Size([100, 32, 50])
print(h_n.size()) #torch.Size([2, 32, 50])


# LSTM
lstm = nn.LSTM(input_size = 20,hidden_size=50,num_layers=2)
lstm_out,(h_n,c_n) = lstm(toy_input)
print("LSTM out:")
print(lstm_out.size())
print(h_n.size())
print(c_n.size())
