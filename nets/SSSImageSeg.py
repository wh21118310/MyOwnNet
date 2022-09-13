# encoding:utf-8
"""
    @Author: DorisFawkes
    @File:
    @Date: 2022/09/13 23:27
    @Description: 
"""
from torch import nn


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()