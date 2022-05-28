# -*- coding: utf-8 -*-

"""
@Time : 2022/5/23
@Author : FaweksLee
@File : transform
@Description : 
"""
from albumentations import Resize, Compose, Normalize

# transforms = A.Compose([A.Resize(256, 256), ToTensorV2()])
from albumentations.pytorch import ToTensorV2

transforms = Compose([
    Resize(height=256, width=256),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(transpose_mask=True)
])
# from torchvision.transforms import ToTensor
#
# transforms = ToTensor()