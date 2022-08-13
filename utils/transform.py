# -*- coding: utf-8 -*-

"""
@Time : 2022/5/23
@Author : FaweksLee
@File : transform
@Description : 
"""
from albumentations import Resize, Compose, Normalize, HorizontalFlip, RandomCrop, GaussNoise, RandomResizedCrop, OneOf, \
    IAAAdditiveGaussianNoise, VerticalFlip, MotionBlur, MedianBlur, Blur, ShiftScaleRotate, RandomBrightnessContrast

# transforms = A.Compose([A.Resize(256, 256), ToTensorV2()])
from albumentations.pytorch import ToTensorV2

transforms_train = Compose([
    Resize(height=256, width=256),
    # RandomResizedCrop(width=224, height=224),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    OneOf([  # 随机选择其中一种
        IAAAdditiveGaussianNoise(),
        GaussNoise(),
    ], p=0.2),
    OneOf([
        MotionBlur(p=0.5),  # 用随即大小的内核将运动模糊应用于输入图像，
        MedianBlur(blur_limit=3, p=0.1),  # 中值滤波
        Blur(blur_limit=3, p=0.1),  # 使用随即大小的内核模糊输入图像。
    ], p=0.2),
    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    RandomBrightnessContrast(p=0.2),
    # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(transpose_mask=True)
])
transforms_valid = Compose([
    Resize(height=256, width=256),
    # HorizontalFlip(p=0.5),
    # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(transpose_mask=True)
])
# from torchvision.transforms import ToTensor
#
# transforms = ToTensor()
