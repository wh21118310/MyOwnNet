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
        self.relu = nn.ReLU(inplace=True)
        # self.tem_channels = in_channels // 4  # 原版未使用此形式，而是输出通道为out_channels
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1)
        )
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
        self.conv_cat = BasicConv2d(4 * out_channels, out_channels, 3, padding=1)
        self.conv_res = BasicConv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
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
        self.map = nn.Conv2d(self.channel, 1, 7, 1, 3)

    def forward(self, x):
        cab = self.cab(x)
        sab = self.sab(cab)
        map = self.map(sab)

        return sab, map


###################################################################
# ################## Context Exploration Block ####################
###################################################################
class Context_Exploration_Block(nn.Module):
    def __init__(self, input_channels):
        super(Context_Exploration_Block, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)

        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        # self.p1 = nn.Sequential(
        #     nn.Conv2d(in_channels=self.channels_single, out_channels=self.channels_single, kernel_size=1, stride=1,
        #               padding=0),
        #     nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels_single, out_channels=self.channels_single, kernel_size=(3, 1), stride=1,
                      padding=(1, 0), dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(),
            nn.Conv2d(in_channels=self.channels_single, out_channels=self.channels_single, kernel_size=(1, 3),
                      stride=1, padding=(0, 1), dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        # self.p2 = nn.Sequential(
        #     nn.Conv2d(in_channels=self.channels_single, out_channels=self.channels_single, kernel_size=3, stride=1,
        #               padding=1),
        #     nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels_single, out_channels=self.channels_single, kernel_size=(3, 1), stride=1,
                      padding=(2, 0), dilation=2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(),
            nn.Conv2d(in_channels=self.channels_single, out_channels=self.channels_single, kernel_size=(1, 3),
                      stride=1, padding=(0, 2), dilation=2)
        )

        # self.p3 = nn.Sequential(
        #     nn.Conv2d(in_channels=self.channels_single, out_channels=self.channels_single, kernel_size=5, stride=1,
        #               padding=2),
        #     nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels_single, out_channels=self.channels_single, kernel_size=(3, 1), stride=1,
                      padding=(4, 0), dilation=4),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(),
            nn.Conv2d(in_channels=self.channels_single, out_channels=self.channels_single, kernel_size=(1, 3), stride=1, padding=(0, 4), dilation=4),
            nn.BatchNorm2d(self.channels_single), nn.ReLU()
        )

        # self.p4 = nn.Sequential(
        #     nn.Conv2d(in_channels=self.channels_single, out_channels=self.channels_single, kernel_size=7, stride=1,
        #               padding=3),
        #     nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels_single, out_channels=self.channels_single, kernel_size=(3, 1), stride=1, padding=(8, 0), dilation=8),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(),
            nn.Conv2d(in_channels=self.channels_single, out_channels=self.channels_single, kernel_size=(1, 3), stride=1, padding=(0, 8), dilation=8),
            nn.BatchNorm2d(self.channels_single), nn.ReLU()
        )

        self.fusion = nn.Sequential(nn.Conv2d(self.input_channels * 2, self.input_channels, 1, 1, 0),
                                    nn.BatchNorm2d(self.input_channels), nn.ReLU())

    def forward(self, x):
        p1_input = self.p1_channel_reduction(x)
        p1 = self.p1(p1_input)
        # p1_dc = self.p1_dc(p1)

        p2_input = self.p2_channel_reduction(x) + p1
        p2 = self.p2(p2_input)
        # p2_dc = self.p2_dc(p2)

        p3_input = self.p3_channel_reduction(x) + p2
        p3 = self.p3(p3_input)
        # p3_dc = self.p3_dc(p3)

        p4_input = self.p4_channel_reduction(x) + p3
        p4 = self.p4(p4_input)
        # p4_dc = self.p4_dc(p4)

        ce = self.fusion(torch.cat((p1, p2, p3, p4, x), 1))

        return ce


###################################################################
# ######################## Focus Module ###########################
###################################################################

class Focus(nn.Module):
    def __init__(self, channel1, channel2):
        super(Focus, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
                                nn.BatchNorm2d(self.channel1), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))

        self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())
        # self.input_map = nn.UpsamplingBilinear2d(scale_factor=2)
        self.output_map = nn.Sequential(nn.Conv2d(self.channel1, 1, 7, 1, 3),
                                        nn.ReLU(inplace=True))
        self.fp = Context_Exploration_Block(self.channel1)
        self.fn = Context_Exploration_Block(self.channel1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.bn1 = nn.BatchNorm2d(self.channel1)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(self.channel1)
        self.relu2 = nn.ReLU()

    def forward(self, x, y, in_map):
        # x; current-level features
        # y: higher-level features
        # in_map: higher-level prediction

        up = self.up(y)

        input_map = self.input_map(in_map)
        f_feature = x * input_map
        b_feature = x * (1 - input_map)

        fp = self.fp(f_feature)
        fn = self.fn(b_feature)

        refine1 = up - (self.alpha * fp)
        refine1 = self.bn1(refine1)
        refine1 = self.relu1(refine1)

        refine2 = refine1 + (self.beta * fn)
        refine2 = self.bn2(refine2)
        refine2 = self.relu2(refine2)

        output_map = self.output_map(refine2)

        return refine2, output_map


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
    def __init__(self, model_path=None, bk='resnet50', num_classes=1):
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
            self.positioning = Positioning(512)
            # focus
            self.focus3 = Focus(256, 512)
            self.focus2 = Focus(128, 256)
            self.focus1 = Focus(64, 128)
        elif bk in backbone_names2:
            # channel reduction
            self.cr4 = TextureEnhancedModule(1024, 256)
            self.cr3 = TextureEnhancedModule(512, 128)
            self.cr2 = TextureEnhancedModule(256, 64)
            self.cr1 = TextureEnhancedModule(128, 32)
            # positioning
            self.positioning = Positioning(256)
            # focus
            self.focus3 = Focus(128, 256)
            self.focus2 = Focus(64, 128)
            self.focus1 = Focus(32, 64)
        self.testlayer1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.testlayer2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.testlayer3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.testlayer4 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        # x: [batch_size, channel=3, h, w]
        layer1, layer2, layer3, layer4 = self.backbone(x, self.model)
        layer1 = self.testlayer1(layer1)
        layer2 = self.testlayer2(layer2)
        layer3 = self.testlayer3(layer3)
        layer4 = self.testlayer4(layer4)
        predict4 = F.interpolate(layer4, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.interpolate(layer3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(layer2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(layer1, size=x.size()[2:], mode='bilinear', align_corners=True)
        return predict4, predict3, predict2, predict1
        # channel reduction


if __name__ == '__main__':
    data = torch.rand((4, 3, 512, 512)).cuda()
    # net = PFNet(bk='swinT_base').cuda()
    # net = PFNet()
    net = PFNet().cuda()
    result = net(data)
    print(result)
