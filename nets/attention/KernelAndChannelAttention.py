# -*- coding: utf-8 -*-

"""
    @Time : 2022/8/28 0:23
    @Author : FaweksLee
    @Email : 121106010719@njust.edu.cn
    @File : Multi-Attention-Network
    @Description : Copy and modification of the paper in Cite
    @Cite: Li, R., Zheng, S., Duan, C., Zhang, C., Su, J., & Atkinson, P. M. (2021), "Multiattention
Network for Semantic Segmentation of Fine-Resolution Remote Sensing Images," in IEEE Transactions on Geoscience and
Remote Sensing, doi: 10.1109/TGRS.2021.3093977.
"""
import torch.nn.functional as F
import torch
from torchvision import models
from torch.nn import Module, Parameter, Softmax, Conv2d
from torch import nn


def Conv3x3_pad_relu(in_planes, out_planes, kernel_size=None, stride=None, padding=None):
    # 3x3 Conv with padding and relu
    if kernel_size is None:
        kernel_size = 3
    assert isinstance(kernel_size, (int, tuple)), 'kernel_size is not in (int, tuple)!'

    if stride is None:
        stride = 1
    assert isinstance(stride, (int, tuple)), 'stride is not in (int, tuple)!'

    if padding is None:
        padding = 1
    assert isinstance(padding, (int, tuple)), 'stride is not in (int, tuple)!'

    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, bias=True),
        nn.ReLU(inplace=True)
    )


class KernelAttentionModule(Module):
    def __init__(self, in_planes, scale=8, eps=1e-6):
        super(KernelAttentionModule, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_channels = in_planes
        self.softplus = F.softplus
        self.eps = eps

        self.query = Conv2d(in_channels=self.in_channels, out_channels=self.in_channels // scale, kernel_size=1)
        self.key = Conv2d(in_channels=self.in_channels, out_channels=self.in_channels // scale, kernel_size=1)
        self.value = Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        b, c, h, w = x.shape
        Q = self.query(x).view(b, -1, h * w)
        K = self.key(x).view(b, -1, h * w)
        V = self.value(x).view(b, -1, h * w)

        Q = self.softplus(Q).permute(0, 2, 1)
        K = self.softplus(K)

        KV = torch.einsum("bmn, bcn -> bmc", K, V)  # 爱因斯坦求和约定，与eniops相比，后者主要用于变形

        norm = 1 / torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps)  # 按照最后一维进行汇总，bcm->bc

        # weight_value
        weight_value = torch.einsum("bnm, bmc, bn-> bcn", Q, KV, norm)  # Dont Understand
        weight_value = weight_value.view(b, c, h, w)

        return (x + self.gamma * weight_value).contiguous()


class ChannelAttentionModule(nn.Module):
    def __init__(self):
        super(ChannelAttentionModule, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.shape
        proj_query = x.view(b, c, -1)
        proj_key = x.view(b, c, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)  # 元素点乘, b c c
        energy_new = torch.max(energy, -1, keepdim=True)[0]  # 从每行中得到最大值，返回其值和对应索引,[0]只要值
        energy_new = energy_new.expand_as(energy) - energy

        attention = self.softmax(energy_new)
        proj_value = x.view(b, c, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(b, c, h, w)
        out = self.gamma * out
        return out


class KAC_Attention(Module):
    def __init__(self, in_planes):
        super(KAC_Attention, self).__init__()
        self.KAM = KernelAttentionModule(in_planes)
        self.CAM = ChannelAttentionModule()

    def forward(self, x):
        return self.KAM(x) + self.CAM(x)


class DecoderBlock(Module):
    def __init__(self, in_planes, n_filters):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_planes
        self.out_channels = n_filters
        self.Conv = nn.Sequential(
            Conv2d(in_channels=self.in_channels, out_channels=self.in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(self.in_channels // 4),
            nn.ReLU(inplace=True)
        )
        self.DeConv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.in_channels // 4, out_channels=self.in_channels // 4, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.in_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.DeConv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels // 4, out_channels=self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.Conv(x)
        out = self.DeConv2(out)
        return self.DeConv3(out)


class MultiAttentionNet(Module):
    def __init__(self, in_channels=3, num_classes=3, filters=(256, 512, 1024, 2048)):
        super(MultiAttentionNet, self).__init__()
        self.name = "MANet"
        self.channels = filters
        backbone = models.resnet50(pretrained=True)  # according to the paper
        self.firstConv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)

        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        self.attention1 = KAC_Attention(self.channels[0])
        self.attention2 = KAC_Attention(self.channels[1])
        self.attention3 = KAC_Attention(self.channels[2])
        self.attention4 = KAC_Attention(self.channels[3])

        self.decoder1 = DecoderBlock(self.channels[0], self.channels[0])
        self.decoder2 = DecoderBlock(self.channels[1], self.channels[0])
        self.decoder3 = DecoderBlock(self.channels[2], self.channels[1])
        self.decoder4 = DecoderBlock(self.channels[3], self.channels[2])

        self.finalConv = nn.Sequential(
            nn.ConvTranspose2d(self.channels[0], 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 3, padding=1)
        )

    def forward(self, x):
        x1 = self.firstConv(x)
        e1 = self.encoder1(x1)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        at4 = self.attention4(e4)
        at3 = self.attention3(e3)
        at2 = self.attention2(e2)
        at1 = self.attention1(e1)

        # Decoder
        d4 = self.decoder4(at4) + at3 # 提前element-wise add
        d3 = self.decoder3(d4) + at2
        d2 = self.decoder2(d3) + at1
        d1 = self.decoder1(d2)

        out = self.finalConv(d1)
        return out

if __name__ == '__main__':
    batch, channels, height, width = 4, 3, 512, 512
    x = torch.randn(batch, channels, height, width)
    net = MultiAttentionNet(3, 3)
    out = net(x)
    print(out.shape)




