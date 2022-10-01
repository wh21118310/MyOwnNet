# -*- coding: utf-8 -*-

"""
    @Time : 2022/9/29 22:14
    @Author : FaweksLee
    @Email : 121106010719@njust.edu.cn
    @File : MRDDANet
    @Description : 
"""
import torch
from torch import nn


class MSBlock(nn.Module):
    def __init__(self, in_channels):
        super(MSBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2, stride=1)
        self.conv2_3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, padding=3, stride=1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x2_3 = self.conv2_3(x)
        x = torch.cat([x2_1, x2_2, x2_3], dim=1)
        x = self.conv3(x)
        return x


class CABlock(nn.Module):
    def __init__(self, in_channels):
        super(CABlock, self).__init__()
        self.avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=2 * in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=2 * in_channels, out_channels=in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.avg(x)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x = torch.mul(x1, x)
        return x


class PABlock(nn.Module):
    def __init__(self, in_channels):
        super(PABlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels // 2), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=int(in_channels // 2), out_channels=in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.conv(x)
        x = torch.mul(x1, x)
        return x


class RDDAB(nn.Module):
    def __init__(self, in_channels):
        super(RDDAB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.PReLU(num_parameters=1, init=0.25),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        )
        self.conv2 = nn.Sequential(
            nn.PReLU(num_parameters=1, init=0.25),
            nn.Conv2d(in_channels=2 * in_channels, out_channels=in_channels, kernel_size=5, padding=2),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        )
        self.conv3 = nn.Sequential(
            nn.PReLU(num_parameters=1, init=0.25),
            nn.Conv2d(in_channels=3 * in_channels, out_channels=in_channels, kernel_size=7, padding=3),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        )
        self.conv4_1 = nn.Conv2d(in_channels=4*in_channels, out_channels=in_channels, kernel_size=1)
        self.CAB = CABlock(in_channels=in_channels)
        self.PAB = PABlock(in_channels=in_channels)
        self.conv4_2 = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x0 = x
        x1 = self.conv1(x)  # 第一个block
        x1 = x1 + x  # 第一个+
        x2 = torch.cat([x1, x], dim=1)  # 第一个cat
        x2 = self.conv2(x2)  # 第二个block
        x2 = x2 + x1  # 第二个+
        x3 = torch.cat([x2, x1, x], dim=1)  # 第二个cat
        x3 = self.conv3(x3)
        x3 = x3 + x2
        x4 = torch.cat([x3, x2, x1, x], dim=1)
        x4_1 = self.conv4_1(x4)
        x4_2CAB = self.CAB(x4_1)
        x4_2PAB = self.PAB(x4_1)
        x4_2 = torch.cat([x4_2CAB, x4_2PAB], dim=1)
        x4_3 = self.conv4_2(x4_2)
        return x0 + x4_3


class MRDDA(nn.Module):
    def __init__(self, in_channels):
        super(MRDDA, self).__init__()
        self.MSB = MSBlock(in_channels=in_channels)
        self.RDDA1 = RDDAB(in_channels=64)
        self.RDDA2 = RDDAB(in_channels=64)
        self.RDDA3 = RDDAB(in_channels=64)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=in_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.MSB(x)
        x1 = self.RDDA1(x1)
        x2 = self.RDDA2(x1)
        x3 = self.RDDA3(x2)
        x3 = torch.cat([x1, x2, x3], dim=1)
        x3 = self.conv4(x3)
        return x3 + x


if __name__ == '__main__':
    data = torch.randn((4, 3, 512, 512))
    model = MRDDA(3)
    result = model(data)
    print(result)
