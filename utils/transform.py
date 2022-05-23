# -*- coding: utf-8 -*-

"""
@Time : 2022/5/23
@Author : FaweksLee
@File : transform
@Description : 
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
train_transform = A.Compose([
    A.Resize(256, 256),
    ToTensorV2()
])
val_transform = A.Compose([
    A.Resize(256, 256),
    ToTensorV2()
])
test_transform = A.Compose([
    A.Resize(256, 256),
    ToTensorV2()
])