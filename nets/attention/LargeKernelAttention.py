# -*- coding: utf-8 -*-

"""
    @Time : 2022/9/2 21:25
    @Author : FaweksLee
    @Email : 121106010719@njust.edu.cn
    @File : LargeKernelAttention
    @Description : copy from VAN https://github.com/Visual-Attention-Network/VAN-Segmentation
"""
from torch import nn


class LKModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
            nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        )
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv1(attn)
        return attn
