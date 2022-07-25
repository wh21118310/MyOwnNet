# -*- coding: utf-8 -*-

"""
    @Time : 2022/7/15 15:54
    @Author : FaweksLee
    @Email : 121106010719@njust.edu.cn
    @File : PSPNet
    @Description : Pyramid Scene Parsing Network, CVPR2017
                The original code is based on Caffee,and we need to based on pytorch
"""
from torchinfo import summary
from torchvision.models import resnet50, resnet101
from torchvision.models._utils import IntermediateLayerGetter
import torch
import torch.nn as nn
from torch.nn import functional as F


class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
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


class PSPHEAD(nn.Module):
    def __init__(self, in_channels,
                 out_channels, pool_sizes=[1, 2, 3, 6], num_classes=31):
        super(PSPHEAD, self).__init__()
        self.pool_sizes = pool_sizes
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.psp_modules = PPM(self.pool_sizes, self.in_channels, self.out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(self.in_channels + len(self.pool_sizes) * self.out_channels, self.out_channels, kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.psp_modules(x)
        out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out


class Pspnet(nn.Module):
    def __init__(self, num_classes):
        super(Pspnet, self).__init__()
        self.num_classes = num_classes
        self.backbone = IntermediateLayerGetter(
            resnet50(pretrained=False, replace_stride_with_dilation=[False, True, True]),
            return_layers={'layer4': 'stage4'}
        )
        self.decoder = PSPHEAD(in_channels=2048, out_channels=512, pool_sizes=[1, 2, 3, 6],
                               num_classes=self.num_classes)
        self.cls_seg = nn.Sequential(nn.Conv2d(512, self.num_classes, kernel_size=3, padding=1))

    def forward(self, x):
        _, _, h, w = x.size()
        feats = self.backbone(x)
        x = self.decoder(feats["stage4"])
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        x = self.cls_seg(x)
        return x


if __name__ == "__main__":
    # model = Pspnet(num_classes=3)
    # model = model.cuda()
    # a = torch.rand([2, 3, 224, 224])
    # a = a.cuda()
    # print(model(a).shape)
    model = resnet50(pretrained=False).cuda()
    data = torch.rand([2, 3, 512, 512])
    result = summary(model, data.size())
    print(result)

