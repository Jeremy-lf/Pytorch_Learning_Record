# -*-coding:utf-8-*-
import numpy as np
import  torch

# import cv2
from torch import nn,optim
from torch.autograd import Variable

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

#numpy to tensor
x_train = torch.from_numpy(x_train)

y_train = torch.from_numpy(y_train)


#model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # input and output is 1 dimension

    def forward(self, x):
        out = self.linear(x)
        return out
model = LinearRegression()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-4)

num_epochs = 10000
for epoch in range(num_epochs):
    inputs = Variable(x_train)
    target = Variable(y_train)
    model.train()
    # forward
    out = model(inputs) # 前向传播
    loss = criterion(out, target) # 计算loss
    # backward
    optimizer.zero_grad() # 梯度归零
    loss.backward() # 方向传播
    optimizer.step() # 更新参数
    print(loss)
    print(loss.data)
    print(loss.item())#Use torch.Tensor.item() to get a Python number from a tensor containing a single value:

    if (epoch+1) % 20 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'.format(epoch+1,
                                                  num_epochs,
                                                  loss.data))



if torch.cuda.is_available():
    mdoel = model.cuda()


model.eval()
predict = model(x_train)
predict = predict.data.numpy()
print(predict)


# torch.save(model.state_dict(), './linear.pth')

