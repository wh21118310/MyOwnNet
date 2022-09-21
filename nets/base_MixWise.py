# -*- coding: utf-8 -*-

"""
    @Time : 2022/9/16 16:07
    @Author : FaweksLee
    @Email : 121106010719@njust.edu.cn
    @File : base_MixWise
    @Description : 
"""
import torch
from torch import nn
import torch.nn.functional as F

from nets.backbone.bk import Backbone


###################################################################
# ################## TEM Block ###################################
###################################################################
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class TextureEnhancedModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(TextureEnhancedModule, self).__init__()
        # self.tem_channels = in_channels // 4  # 原版未使用此形式，而是输出通道为out_channels
        # self.branch0 = BasicConv2d(in_channels, out_channels, 1)
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channels, out_channels, kernel_size=3, groups=out_channels, padding=1),
            # BasicConv2d(out_channels, out_channels, kernel_size=7, padding=9, groups=out_channels, dilation=3),
            BasicConv2d(out_channels, out_channels, kernel_size=1),
            BasicConv2d(out_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channels, out_channels, kernel_size=5, groups=out_channels, padding=2),
            # BasicConv2d(out_channels, out_channels, kernel_size=7, padding=9, groups=out_channels, dilation=3),
            BasicConv2d(out_channels, out_channels, kernel_size=1),
            BasicConv2d(out_channels, out_channels, kernel_size=3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channels, out_channels, kernel_size=7, groups=out_channels, padding=3),
            # BasicConv2d(out_channels, out_channels, kernel_size=7, padding=9, groups=out_channels, dilation=3),
            BasicConv2d(out_channels, out_channels, kernel_size=1),
            BasicConv2d(out_channels, out_channels, kernel_size=3, padding=7, dilation=7)
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 9), padding=(0, 4)),
            BasicConv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0)),
            BasicConv2d(out_channels, out_channels, kernel_size=9, groups=out_channels, padding=4),
            # BasicConv2d(out_channels, out_channels, kernel_size=7, padding=9, groups=out_channels, dilation=3),
            BasicConv2d(out_channels, out_channels, kernel_size=1),
            BasicConv2d(out_channels, out_channels, kernel_size=3, padding=9, dilation=9)
        )
        self.conv_cat = BasicConv2d(4 * out_channels, out_channels, 3, padding=1)
        self.conv_res = BasicConv2d(in_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x_cat = self.conv_cat(torch.cat((x4, x1, x2, x3), 1) + x)
        x = self.relu(x_cat)
        return x


backbone_names1 = [  # C: 2048,1024,512,256
    'resnet50', 'res2net50',
]
backbone_names2 = [  # C: 1024,512,256,128
    'convnext_base', 'swinT_base'
]


class PFNet(nn.Module):
    def __init__(self, model_path=None, bk='resnet50'):
        super(PFNet, self).__init__()
        # params
        self.model = bk
        # backbone
        self.backbone = Backbone(bk=bk, model_path=model_path)

        if bk in backbone_names1:
            # channel reduction
            self.cr4 = TextureEnhancedModule(2048, 512)
            self.cr3 = TextureEnhancedModule(1024, 256)
            self.cr2 = TextureEnhancedModule(512, 128)
            self.cr1 = TextureEnhancedModule(256, 64)
            self.testlayer1 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
                nn.BatchNorm2d(1),
                nn.ReLU(True)
            )
            self.testlayer2 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1),
                nn.BatchNorm2d(1),
                nn.ReLU(True)
            )
            self.testlayer3 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
                nn.BatchNorm2d(1),
                nn.ReLU(True)
            )
            self.testlayer4 = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1),
                nn.BatchNorm2d(1),
                nn.ReLU(True)
            )
        elif bk in backbone_names2:
            # channel reduction
            self.cr4 = TextureEnhancedModule(1024, 256)
            self.cr3 = TextureEnhancedModule(512, 128)
            self.cr2 = TextureEnhancedModule(256, 64)
            self.cr1 = TextureEnhancedModule(128, 32)
            self.testlayer1 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
                nn.BatchNorm2d(1),
                nn.ReLU(True)
            )
            self.testlayer2 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
                nn.BatchNorm2d(1),
                nn.ReLU(True)
            )
            self.testlayer3 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1),
                nn.BatchNorm2d(1),
                nn.ReLU(True)
            )
            self.testlayer4 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
                nn.BatchNorm2d(1),
                nn.ReLU(True)
            )
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        # x: [batch_size, channel=3, h, w]
        layer1, layer2, layer3, layer4 = self.backbone(x, self.model)

        # channel reduction
        cr4 = self.cr4(layer4)
        cr3 = self.cr3(layer3)
        cr2 = self.cr2(layer2)
        cr1 = self.cr1(layer1)
        cr1 = self.testlayer1(cr1)
        cr2 = self.testlayer2(cr2)
        cr3 = self.testlayer3(cr3)
        cr4 = self.testlayer4(cr4)
        predict4 = F.interpolate(cr4, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.interpolate(cr3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(cr2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(cr1, size=x.size()[2:], mode='bilinear', align_corners=True)
        return predict4, predict3, predict2, predict1
