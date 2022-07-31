# encoding:utf-8
"""
    @Author: DorisFawkes
    @File:
    @Date: 2022/07/30 12:20
    @Description: InterlacedSparseSelfAttention
"""
import torch
from torch import nn
from torch.nn import init

from SelfAttention import ScaledDotProductAttention


class InterlacedSparseSelfAttention(nn.Module):

    def __init__(self, P_h, P_w, num_channels):
        super().__init__()
        self.P_h = P_h
        self.P_w = P_w
        self.attention = ScaledDotProductAttention(d_model=num_channels, d_k=num_channels, d_v=num_channels, h=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.size()
        Q_h, Q_w = h // self.P_h, w // self.P_w
        x = x.reshape(b, c, Q_h, self.P_h, Q_w, self.P_w)
        # Long-range Attention
        x = x.permute(0, 3, 5, 1, 2, 4)
        x = x.reshape(b * self.P_h * self.P_w, Q_h*Q_w, c)
        x = self.attention(x, x, x)
        x = x.reshape(b, self.P_h, self.P_w, c, Q_h, Q_w)

        # Short-range Attention
        x = x.permute(0, 4, 5, 3, 1, 2)
        x = x.reshape(b * Q_h * Q_w, self.P_h * self.P_w, c)
        x = self.attention(x, x, x)
        x = x.reshape(b, Q_h, Q_w, c, self.P_h, self.P_w)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)
        return x
if __name__ == '__main__':
    input = torch.randn(50, 128, 7, 7)
    danet = InterlacedSparseSelfAttention(P_h=7,P_w=7, num_channels=128)
    print(danet(input).shape)