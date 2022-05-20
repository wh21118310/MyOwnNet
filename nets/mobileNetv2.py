# -*- coding: utf-8 -*-

"""
@Time : 2022/5/10
@Author : FaweksLee
@File : mobileNetv2
@Description : 
"""
import math

import torch.nn as nn
from torch.nn import BatchNorm2d, Conv2d, ReLU6, Sequential

from utils.data_process import load_url


def conv_bn(input, output, stride):
    return Sequential(
        Conv2d(in_channels=input, out_channels=output, kernel_size=3, stride=stride, padding=1, bias=False),
        BatchNorm2d(num_features=output),
        ReLU6(inplace=True)
    )


def conv_1x1_bn(input, output):
    return Sequential(
        Conv2d(in_channels=input, out_channels=output, kernel_size=1, stride=1, bias=False),
        BatchNorm2d(num_features=output),
        ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    # --------------------------------------------#
    #   模块相较于Residual模块，前者通道数先增后减，后者则先减后增
    # --------------------------------------------#
    def __init__(self, inp, out, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)  # 返回四舍五入值
        self.use_res_connect = self.stride == 1 and inp == out

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # --------------------------------------------#
                #   进行3x3的逐层卷积，进行跨特征点的特征提取
                # --------------------------------------------#
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(3, 3), stride=(stride, stride),
                          padding=(1, 1), groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # -----------------------------------#
                #   利用1x1卷积进行通道数的调整
                # -----------------------------------#
                nn.Conv2d(in_channels=hidden_dim, out_channels=out, kernel_size=(1, 1), stride=(1, 1), bias=False),
                BatchNorm2d(out),
            )
        else:
            self.conv = nn.Sequential(
                # -----------------------------------#
                #   利用1x1卷积进行通道数的上升,保证网络更好的表征能力
                # -----------------------------------#
                nn.Conv2d(in_channels=inp, out_channels=hidden_dim, kernel_size=(1, 1), stride=(1, 1), bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # --------------------------------------------#
                #   进行3x3的逐层卷积，进行跨特征点的特征提取
                # --------------------------------------------#
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(3, 3), stride=(stride, stride),
                          padding=(1, 1), groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # -----------------------------------#
                #   利用1x1卷积进行通道数的下降，减少计算量
                # -----------------------------------#
                nn.Conv2d(in_channels=hidden_dim, out_channels=out, kernel_size=(1, 1), stride=(1, 1), bias=False),
                BatchNorm2d(out),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        inp = 32
        last = 1280
        interverted_residual_setting = [  # 完成三次或四次下采样的过程
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]
        assert input_size % 32 == 0, "input_size is not the times of 32"
        input_channel = int(inp * width_mult)
        self.last_channel = int(last * width_mult) if width_mult > 1.0 else last
        self.features = [conv_bn(3, input_channel, 2)]
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.size()
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv2(pretrained=False, **kwargs):
    model = MobileNetV2(n_class=1000, **kwargs)
    if pretrained:
        model.load_state_dict(load_url('https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0'
                                       '/mobilenet_v2.pth.tar'),
                              strict=False)
    return model


if __name__ == '__main__':
    model = mobilenetv2()
    for i, layer in enumerate(model.features):
        print(i, layer)
