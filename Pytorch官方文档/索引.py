# -*- coding: utf-8 -*-
"""
Created on : 2021/3/17 13:19

@author: Jeremy
"""
import torch

a = torch.Tensor(4,5)
print(a)
print(a[0:1,:2])
print(a[0,:2])  # 注意和前一种索引出来的值相同，shape不同
print(a[[1,2]])  # 容器索引

'''
3.3845e+15  0.0000e+00  3.3846e+15  0.0000e+00  3.3845e+15
 0.0000e+00  3.3845e+15  0.0000e+00  3.3418e+15  0.0000e+00
 3.3845e+15  0.0000e+00  3.3846e+15  0.0000e+00  0.0000e+00
 0.0000e+00  1.5035e+38  8.5479e-43  1.5134e-43  1.2612e-41
[torch.FloatTensor of size 4x5]


 3.3845e+15  0.0000e+00
[torch.FloatTensor of size 1x2]


 3.3845e+15
 0.0000e+00
[torch.FloatTensor of size 2]


 0.0000e+00  3.3845e+15  0.0000e+00  3.3418e+15  0.0000e+00
 3.3845e+15  0.0000e+00  3.3846e+15  0.0000e+00  0.0000e+00
[torch.FloatTensor of size 2x5]
'''