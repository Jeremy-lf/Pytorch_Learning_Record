# -*- coding: utf-8 -*-
"""
Created on : 2021/4/25 17:33

@author: Jeremy
"""
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self):
        super(PatchEmbed,self).__init__()
        self.proj = nn.Conv2d(3, 3, kernel_size=2, stride=2)

    def forward(self, x):
        B, C, H, W = x.shape # x = (2,3,12,12)
        x = self.proj(x) #torch.Size([2, 3, 6, 6])
        # print("x1:",x.shape)
        x = x.flatten(2)  # torch.Size([2, 3, 36])
        # print("x2:",x.shape)
        x = x.transpose(1, 2)  # B Ph*Pw C # torch.Size([2, 36, 3])
        # print("x2:",x.shape)
        # flatten(2)等于从2维度开始进行展平操作，x的维度为b c h w ，
        # 设patch_size为4,则结果为 b (h/4)*(w/4) 16*c
        #
        # if self.norm is not None:
        #     x = self.norm(x)
        return x


# 整体结构中，通过PatchEmbed()分割出图像块，再经过相应层数的BasicLayer()。
class SwinTransformer(nn.Module):
    def __init__(self):
        super(SwinTransformer,self).__init__()

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed()

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer()
            self.layers.append(layer)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # 用于输出的分类维度，可以根据自己的需要更改

    def forward_features(self, x):
        x = self.patch_embed(x)
        # b h w c -> b (h/4)*(w/4) 16*c
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        # 以下用于进行分类等任务，可以根据需要进行调整
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class BasicLayer(nn.Module):

    def __init__(self):

        super(BasicLayer,self).__init__()
        # build blocks
        self.blocks = nn.ModuleList([SwinTransformerBlock() for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample()
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
              x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
            #x:  b h w c -> b h/2*w/2 2c
        print('basic_layer:{}'.format(x.shape))
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self.input_resolution = input_resolution


    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        print("x0:",x.shape)
        x = x.view(-1, 4 * C)  # B H/2*W/2 4*C
        print("x1:",x.shape)
        x = self.norm(x)
        print("x2:",x.shape)
        x = self.reduction(x)
        print("x3:",x.view(B,-1,2*C).shape)
        #通过reduction，x: b h/2*w/2 2c,即实现了降维的操作，并降低了宽高的特征大小
        return x

input  = torch.randn(2,16*16,3)
model = PatchMerging([16,16],3)
# print(model(input))
output = model(input)
x = output.view(2,-1,6)
model1 = PatchMerging([8,8],6)
print(model1(x))
