# -*- coding: utf-8 -*-

"""
    @Time : 2022/8/21 15:19
    @Author : FaweksLee
    @Email : 121106010719@njust.edu.cn
    @File : test_model
    @Description : 
"""
from segmentation_models_pytorch import DeepLabV3Plus
from torch import nn
from torch.cuda.amp import autocast


class test_mo(nn.Module):
    def __init__(self, model_name, n_class):
        super().__init__()
        self.model = DeepLabV3Plus(encoder_name=model_name, encoder_weights='imagenet', in_channels=3, classes=n_class)

    @autocast()
    def forward(self, x):
        result = self.model(x)
        return result
