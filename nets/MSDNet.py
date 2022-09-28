# -*- coding: utf-8 -*-

"""
    @Time : 2022/9/27 8:26
    @Author : FaweksLee
    @Email : 121106010719@njust.edu.cn
    @File : MSDNet
    @Description : The copied version of paper "HYPERSPECTRAL IMAGE DENOISING BASED ON MULTI-STREAM DENOISING NETWORK"
"""
import torch
from segmentation_models_pytorch import Unet
from torch import nn
from torch.nn.functional import interpolate


class multiScale(nn.Module):
    def __init__(self, in_channels, kernel):
        super(multiScale, self).__init__()
        pad = int(kernel // 2)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=kernel, stride=1, padding=pad),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel, stride=1, padding=pad),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel, stride=1, padding=pad),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel, stride=1, padding=pad),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=kernel, stride=1, padding=pad),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=kernel, stride=1, padding=pad),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.block1(x)  # 16
        x2 = self.block2(x1)  # 32
        x3 = self.block3(x2)  # 64
        x4 = self.block4(x3)  # 64
        x5 = self.block5(x4)  # 32
        x6 = self.block6(x5)  # 16
        return torch.cat([x1, x2, x3, x4, x5, x6], dim=1)


class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        # self.pool_sizes = pool_sizes
        # self.in_channels = in_channels
        # self.out_channels = out_channels

        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveMaxPool2d(pool_size),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                )
            )

    def forward(self, x):
        out_puts = []
        for ppm in self:
            ppm_out = interpolate(ppm(x), size=x.size()[-2:], mode='bilinear', align_corners=True)
            out_puts.append(ppm_out)
        return out_puts


class NoiseEstimation(nn.Module):
    def __init__(self, in_channels):
        super(NoiseEstimation, self).__init__()
        self.ms1 = multiScale(in_channels=in_channels, kernel=3)
        self.ms2 = multiScale(in_channels=in_channels, kernel=5)
        self.ms3 = multiScale(in_channels=in_channels, kernel=7)
        self.PPM = PPM([1, 2, 3, 6], in_channels=672, out_channels=4 * in_channels)
        self.GMP = nn.AdaptiveMaxPool2d((1, 1))
        self.FC = nn.Sequential(
            nn.Linear(16 * in_channels, in_channels),
            nn.Linear(in_channels, 16 * in_channels),
        )
        self.down = nn.Conv2d(in_channels=16 * in_channels, out_channels=in_channels, kernel_size=1)

    def forward(self, x):
        ms1 = self.ms1(x)
        ms2 = self.ms2(x)
        ms3 = self.ms3(x)
        ms_in = torch.cat([ms1, ms2, ms3], dim=1)
        ms_in = self.PPM(ms_in)
        ms_in = torch.cat(ms_in, dim=1)
        ms_fc = self.GMP(ms_in)
        ms_fc = ms_fc.view(ms_fc.size(0), -1)
        ms_fc = self.FC(ms_fc)
        ms_fc = ms_fc.view(ms_fc.size(0), -1, 1, 1)
        result = self.down(torch.mul(ms_in, ms_fc))
        return result


class MSDNet(nn.Module):
    def __init__(self, channels):
        super(MSDNet, self).__init__()
        self.estimate = NoiseEstimation(in_channels=channels)
        self.Unet = Unet(in_channels=channels, classes=channels)
        self.relu = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        est = self.estimate(x)
        mid_x = x + est
        result = self.Unet(mid_x)
        result = result + x
        result = self.relu(result)
        return result


if __name__ == '__main__':
    data = torch.randn((4, 3, 512, 512))
    # model = NoiseEstimation(3).cuda()
    model = MSDNet(3)
    result = model(data)
    print(result)
