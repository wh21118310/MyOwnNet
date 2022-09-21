# -*- coding: utf-8 -*-

"""
    @Time : 2022/7/23 17:22
    @Author : FaweksLee
    @Email : 121106010719@njust.edu.cn
    @File : PFNet_ASPP_Depthwise
    @Description : Inspired by PFNet & Deeplabv3
"""
import torch
import torch.nn as nn
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


###################################################################
# ################## Channel Attention Block ######################
###################################################################
class CA_Block(nn.Module):
    def __init__(self, in_dim):
        super(CA_Block, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : channel attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


###################################################################
# ################## Spatial Attention Block ######################
###################################################################
class SA_Block(nn.Module):
    def __init__(self, in_dim):
        super(SA_Block, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : spatial attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


###################################################################
# ##################### Positioning Module ########################
###################################################################
class Positioning(nn.Module):
    def __init__(self, channel):
        super(Positioning, self).__init__()
        self.channel = channel
        self.cab = CA_Block(self.channel)
        self.sab = SA_Block(self.channel)
        # self.map = nn.Conv2d(self.channel, self.channel, 7, 1, 3)

    def forward(self, x):
        cab = self.cab(x)
        sab = self.sab(cab)
        # map = self.map(sab)

        return sab


###################################################################
###################   UpSample Connection  ########################
###################################################################



###################################################################
# ########################## NETWORK ##############################
###################################################################
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

            # positioning
            self.positioning4 = Positioning(512)
            self.positioning3 = Positioning(256)
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
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        # x: [batch_size, c, h, w]
        layer1, layer2, layer3, layer4 = self.backbone(x, self.model)

        # channel reduction
        cr4 = self.cr4(layer4)  # [b,2048,16,16]->[b, 512, 16, 16]
        cr3 = self.cr3(layer3)  # [b, 1024, 32, 32] -> [b, 256, 32, 32]
        cr2 = self.cr2(layer2)  # [b, 512, 64, 64] -> [b, 128, 64, 64]
        cr1 = self.cr1(layer1)  # [b, 256, 128, 128]->[b, 64, 128, 128]

        # positioning
        positioning4 = self.positioning4(cr4)
        positioning3 = self.positioning3(cr3)
        layer1 = self.testlayer1(cr1)
        layer2 = self.testlayer2(cr2)
        layer3 = self.testlayer3(positioning3)
        layer4 = self.testlayer4(positioning4)
        predict4 = F.interpolate(layer4, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.interpolate(layer3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(layer2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(layer1, size=x.size()[2:], mode='bilinear', align_corners=True)
        return predict4, predict3, predict2, predict1


if __name__ == '__main__':
    data = torch.rand((4, 3, 512, 512)).cuda()
    # net = PFNet(bk='swinT_base').cuda()
    # net = PFNet()
    net = PFNet().cuda()
    result = net(data)
    print(result)
