# encoding:utf-8
"""
    @Author: DorisFawkes
    @File:
    @Date: 2022/09/13 23:27
    @Description: 
"""
import torch
from einops import rearrange
from torch import nn


class conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(conv, self).__init__()
        pad = int(kernel_size // 2)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                              padding=pad, bias=False, dilation=dilation)
        self.BN = nn.BatchNorm2d(out_channels)
        self.Relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.BN(x)
        return self.Relu(x)


class DMDC(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(DMDC, self).__init__()
        middle_channels = int(in_channels // 2)
        self.DMDConv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, dilation=dilation,
                      stride=2, padding=dilation),
            nn.Conv2d(in_channels=in_channels, out_channels=middle_channels, kernel_size=1),
            nn.Conv2d(in_channels=middle_channels, out_channels=middle_channels, kernel_size=1),
        )
        self.DMDCPool = nn.Sequential(
            nn.AdaptiveAvgPool2d((dilation, dilation)),
            nn.Conv2d(in_channels=in_channels, out_channels=middle_channels, kernel_size=1),
            nn.Conv2d(in_channels=middle_channels, out_channels=middle_channels, kernel_size=dilation)
        )
        self.Out_conv = nn.Conv2d(in_channels=middle_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.DMDConv(x)
        x2 = self.DMDCPool(x)
        x_out = x1 * x2
        x_out = self.Out_conv(x_out)
        return x_out


class DMDConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DMDConv, self).__init__()
        self.DMDC1 = DMDC(in_channels=in_channels, out_channels=out_channels, dilation=1)
        self.DMDC2 = DMDC(in_channels=in_channels, out_channels=out_channels, dilation=3)
        self.DMDC3 = DMDC(in_channels=in_channels, out_channels=out_channels, dilation=5)
        self.upChannel = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.DMDC1(x)
        x2 = self.DMDC2(x)
        x3 = self.DMDC3(x)
        return self.upChannel(torch.cat([x1, x2, x3], dim=1))


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.DMDC1 = DMDConv(in_channels=64, out_channels=128)
        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.AvgPool2d(kernel_size=2)
        )
        self.DMDC2 = DMDConv(in_channels=128, out_channels=256)
        self.encoder3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2)
        )
        self.DMDC3 = DMDConv(in_channels=256, out_channels=512)
        self.encoder4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, padding=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, padding=2),
            nn.AvgPool2d(kernel_size=2)
        )
        self.DMDC4 = DMDConv(in_channels=512, out_channels=1024)
        self.encoder5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=7, padding=3),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=7, padding=3),
            nn.AvgPool2d(kernel_size=2)
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        d1 = self.DMDC1(e1)
        d1 = d1 + e2
        d2 = self.DMDC2(d1)
        d2 = d2 + e3
        d3 = self.DMDC3(d2)
        d3 = d3 + e4
        d4 = self.DMDC4(d3)
        e5 = d4 + e5
        return e1, e2, e3, e4, e5


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.AveragePool = nn.AdaptiveAvgPool2d(1)
        self.MaxPool = nn.AdaptiveMaxPool2d(1)
        self.conv1x1 = nn.Conv2d(channels * 2, channels, 1)
        self.conv1x1_2 = nn.Conv2d(channels * 3, channels // 2, 1)
        self.conv1x1_3 = nn.Conv2d(channels // 2, channels, 1)
        self.Softmax = nn.Softmax()

    def forward(self, x):
        avg = self.AveragePool(x)
        max = self.MaxPool(x)
        mid = torch.cat([avg, max], dim=1)
        mid = self.conv1x1(mid)
        mid = torch.cat([avg, max, mid], dim=1)
        mid = self.conv1x1_2(mid)
        mid = self.conv1x1_3(mid)
        mid = self.Softmax(mid)
        mid = torch.matmul(mid, avg)
        return mid + max


class ARFM(nn.Module):
    def __init__(self, channels):
        super(ARFM, self).__init__()
        middle_channels = int(channels // 4)
        self.conv1_1 = nn.Conv2d(in_channels=channels, out_channels=middle_channels, kernel_size=1)
        self.branch1_1 = nn.Conv2d(in_channels=middle_channels, out_channels=middle_channels, kernel_size=(1, 3),
                                   stride=(1, 2), dilation=(0, 2), padding=(0, 2))
        self.branch1_2 = nn.Conv2d(in_channels=middle_channels, out_channels=middle_channels, kernel_size=(3, 1),
                                   stride=(2, 1), dilation=(2, 0), padding=(2, 0))
        self.branch2_1 = nn.Conv2d(in_channels=middle_channels, out_channels=middle_channels, kernel_size=(1, 3),
                                   dilation=(0, 3), padding=(0, 3), stride=(1, 2))
        self.branch2_2 = nn.Conv2d(in_channels=middle_channels, out_channels=middle_channels, kernel_size=(3, 1),
                                   dilation=(3, 0), padding=(3, 0), stride=(2, 1))
        self.branch3_1 = nn.Conv2d(in_channels=middle_channels, out_channels=middle_channels, kernel_size=(1, 3),
                                   dilation=(0, 4), padding=(0, 4), stride=(1, 2))
        self.branch3_2 = nn.Conv2d(in_channels=middle_channels, out_channels=middle_channels, kernel_size=(3, 1),
                                   dilation=(4, 0), padding=(4, 0), stride=(2, 1))
        self.branch4_1 = nn.Conv2d(in_channels=middle_channels, out_channels=middle_channels, kernel_size=(1, 3),
                                   dilation=(0, 5), padding=(0, 5), stride=(1, 2))
        self.branch4_2 = nn.Conv2d(in_channels=middle_channels, out_channels=middle_channels, kernel_size=(3, 1),
                                   dilation=(5, 0), padding=(5, 0), stride=(2, 1))
        self.avgPool = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=middle_channels, out_channels=middle_channels, kernel_size=1)
        )
        self.cam = ChannelAttention(3 * channels)
        self.out_c1x1 = nn.Conv2d(3 * channels, channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1_1(x)
        x6 = x  # original
        # branch1
        x1_1 = self.branch1_1(x)
        x1_2 = self.branch1_2(x1_1)
        # branch2
        x2 = x1_1 + x1_2 + x
        x2_1 = self.branch2_1(x2)
        x2_2 = x2_1 + x1_2
        x2_2 = self.branch2_2(x2_2)
        # branch 3
        x3 = x2_1 + x2_2 + x
        x3_1 = self.branch3_1(x3)
        x3_2 = x3_1 + x2_2
        x3_2 = self.branch3_2(x3_2)
        # branch 4
        x4 = x + x3_1 + x3_2
        x4_1 = self.branch4_1(x4)
        x4_2 = x4_1 + x3_2
        x4_2 = self.branch4_2(x4_2)
        # branch 5
        x5 = self.avgPool(x)
        x = torch.cat([x1_2, x2_2, x3_2, x4_2, x5, x6], dim=1)
        x = self.cam(x)
        return self.out_c1x1(x)


class FFAM(nn.Module):
    def __init__(self, high_channels, low_channels, kernel_size):
        super(FFAM, self).__init__()
        pad = int(low_channels // 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=low_channels, out_channels=low_channels // 2, kernel_size=(1, kernel_size),
                      padding=(0, pad)),
            nn.Conv2d(in_channels=low_channels, out_channels=1, kernel_size=(kernel_size, 1), padding=(pad, 0))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=low_channels, out_channels=low_channels // 2, kernel_size=(kernel_size, 1),
                      padding=(pad, 0)),
            nn.Conv2d(in_channels=low_channels, out_channels=1, kernel_size=(1, kernel_size), padding=(0, pad))
        )
        self.GMP = nn.AdaptiveMaxPool2d((1, 1))
        self.Sigmoid = nn.Sigmoid()
        self.downConv1 = nn.Conv2d(in_channels=low_channels, out_channels=low_channels, kernel_size=3)
        self.downConv2 = nn.Conv2d(in_channels=low_channels, out_channels=low_channels, kernel_size=3)
        self.downConv3 = nn.Conv2d(in_channels=low_channels, out_channels=low_channels, kernel_size=3)
        self.downConv4 = nn.Conv2d(in_channels=low_channels, out_channels=low_channels, kernel_size=3)
        self.highConv1 = nn.Conv2d(in_channels=high_channels, out_channels=low_channels, kernel_size=3)
        self.highConv2 = nn.Conv2d(in_channels=high_channels, out_channels=low_channels, kernel_size=3)
        self.highConv3 = nn.Conv2d(in_channels=high_channels, out_channels=low_channels, kernel_size=3)
        self.highConv4 = nn.Conv2d(in_channels=high_channels, out_channels=low_channels, kernel_size=3)



    def forward(self, low, high=None):
        x1 = self.conv1(low)
        x2 = self.conv2(low)
        x1 = x1 + x2
        x3 = self.GMP(low)
        x3 = x1 + x3
        up = self.Sigmoid(x3)  # low feature
        up = low * up
        if high is not None:
            down_1 = self.downConv1()




class OverallStructure(nn.Module):
    def __init__(self, in_channels, num_features):
        super(OverallStructure, self).__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.ARFM1 = ARFM(channels=64)
        self.ARFM2 = ARFM(channels=128)
        self.ARFM3 = ARFM(channels=256)
        self.ARFM4 = ARFM(channels=512)
        self.ARFM5 = ARFM(channels=1024)
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, num_features, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, align_corners=True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, align_corners=True)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, padding=2),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.Upsample(scale_factor=2, align_corners=True)
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=5, padding=2),
            nn.Conv2d(256, 256, kernel_size=5, padding=2),
            nn.Upsample(scale_factor=2, align_corners=True)
        )
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=7, padding=3),
            nn.Conv2d(512, 512, kernel_size=7, padding=3),
            nn.Upsample(scale_factor=2, align_corners=True)
        )
        self.FFAM1 = FFAM()


    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        e1 = self.ARFM1(e1)
        e2 = self.ARFM2(e2)
        e3 = self.ARFM3(e3)
        e4 = self.ARFM4(e4)
        e5 = self.ARFM5(e5)
        d5 = self.decoder5(e5)
        d5_1 = d5 + e4
        d4 = self.decoder4(d5_1)
        d4_1 = d4 + e3
        d3 = self.decoder3(d4_1)
        d3_1 = d3 + e2
        d2 = self.decoder2(d3_1)
        d2_1 = d2 + e1
        d1 = self.decoder1(d2_1)




if __name__ == '__main__':
    data = torch.randn((4, 3, 512, 512))
    data2 = torch.randn((4, 3, 1, 1))
    model = Encoder(3)
    result = model(data)
    print(result)
