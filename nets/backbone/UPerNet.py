# -*- coding: utf-8 -*-

"""
    @Time : 2022/7/10 22:37
    @Author : FaweksLee
    @Email : 121106010719@njust.edu.cn
    @File : UPerNet
    @Comment: ECCV 2018
    @Description : https://openaccess.thecvf.com/content_ECCV_2018/papers/Tete_Xiao_Unified_Perceptual_Parsing_ECCV_2018_paper.pdf
"""
import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary

from nets.backbone.bk import Backbone


class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        for pool_size in pool_sizes:
            self.append(nn.Sequential(
                nn.AdaptiveMaxPool2d(pool_size),
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
            )
            )

    def forward(self, x):
        out_puts = []
        for ppm in self:
            ppm_out = F.interpolate(ppm(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
            out_puts.append(ppm_out)
        return out_puts


class PPMHEAD(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6], num_classes=31):
        super(PPMHEAD, self).__init__()
        self.pool_sizes = pool_sizes
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.psp_modules = PPM(self.pool_sizes, self.in_channels, self.out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(self.in_channels + len(self.pool_sizes) * self.out_channels, 4 * self.out_channels,
                      kernel_size=1),
            nn.BatchNorm2d(4 * self.out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.psp_modules(x)
        out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out


class FPNHEAD(nn.Module):
    def __init__(self, channels=2048):
        super(FPNHEAD, self).__init__()
        self.PPMHead = PPMHEAD(in_channels=2048, out_channels=512)

        self.Conv_fuse1 = nn.Sequential(
            nn.Conv2d(channels // 2, channels // 2, 1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU()
        )
        self.Conv_fuse1_ = nn.Sequential(
            nn.Conv2d(channels // 2 + channels, channels // 2, 1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU()
        )
        self.Conv_fuse2 = nn.Sequential(
            nn.Conv2d(channels // 4, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU()
        )
        self.Conv_fuse2_ = nn.Sequential(
            nn.Conv2d(channels // 2 + channels // 4, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU()
        )

        self.Conv_fuse3 = nn.Sequential(
            nn.Conv2d(channels // 8, channels // 8, 1),
            nn.BatchNorm2d(channels // 8),
            nn.ReLU()
        )
        self.Conv_fuse3_ = nn.Sequential(
            nn.Conv2d(channels // 4 + channels // 8, channels // 8, 1),
            nn.BatchNorm2d(channels // 8),
            nn.ReLU()
        )

        self.fuse_all = nn.Sequential(
            nn.Conv2d(channels * 2 - channels // 8, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU()
        )

    def forward(self, input_fpn):
        x1 = self.PPMHead(input_fpn[-1])

        x = F.interpolate(x1, size=(x1.size(2) * 2, x1.size(3) * 2), mode='bilinear', align_corners=True)
        x = torch.cat([x, self.Conv_fuse1(input_fpn[-2])], dim=1)
        x2 = self.Conv_fuse1_(x)

        x = F.interpolate(x2, size=(x2.size(2) * 2, x2.size(3) * 2), mode='bilinear', align_corners=True)
        x = torch.cat([x, self.Conv_fuse2(input_fpn[-3])], dim=1)
        x3 = self.Conv_fuse2_(x)

        x = F.interpolate(x3, size=(x3.size(2) * 2, x3.size(3) * 2), mode='bilinear', align_corners=True)
        x = torch.cat([x, self.Conv_fuse3(input_fpn[-4])], dim=1)
        x4 = self.Conv_fuse3_(x)

        x1 = F.interpolate(x1, x4.size()[-2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, x4.size()[-2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, x4.size()[-2:], mode='bilinear', align_corners=True)

        x = self.fuse_all(torch.cat([x1, x2, x3, x4], 1))

        return x


class UPerNet(nn.Module):
    def __init__(self, num_classes):
        super(UPerNet, self).__init__()
        self.num_classes = num_classes
        self.backbone = Backbone(bk="resnet50")
        self.in_channels = 2048
        self.channels = 512
        self.decoder = FPNHEAD()
        self.cls_seg = nn.Sequential(
            nn.Conv2d(512, self.num_classes, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = list(self.backbone(x))
        x = self.decoder(x)
        x = F.interpolate(x, size=(x.size(2) * 4, x.size(3) * 4), mode='bilinear', align_corners=True)
        x = self.cls_seg(x)
        return x


if __name__ == '__main__':
    data = torch.rand((4, 3, 512, 512)).cuda()
    backbone = UPerNet(3).cuda()
    result = summary(backbone, data.size())
    print(result)