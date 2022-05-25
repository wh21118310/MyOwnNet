# -*- coding: utf-8 -*-

"""
@Time : 2022/5/23
@Author : FaweksLee
@File : transform
@Description : 
"""
from albumentations import Resize, Compose

# transforms = A.Compose([A.Resize(256, 256), ToTensorV2()])
from albumentations.pytorch import ToTensorV2

transforms = Compose([
    # Resize(height=256, width=256),
    ToTensorV2()
])
# from torchvision.transforms import ToTensor
#
# transforms = ToTensor()