# -*- coding: utf-8 -*-

"""
@Time : 2022/5/23
@Author : FaweksLee
@File : transform
@Description : 
"""
import random

# from albumentations import (Resize, Compose, Normalize, HorizontalFlip, RandomCrop, GaussNoise, RandomResizedCrop, \
#                             OneOf, VerticalFlip, MotionBlur, MedianBlur, Blur, ShiftScaleRotate,
#                             RandomBrightnessContrast, \
#                             ISONoise, GaussianBlur)
from PIL import Image
from torchvision.transforms import transforms
# from albumentations.pytorch import ToTensorV2


#
# transforms_trainval = Compose([
#     Resize(height=512, width=512),
#     # RandomResizedCrop(width=224, height=224),
#     HorizontalFlip(p=0.5),
#     VerticalFlip(p=0.5),
#     OneOf([  # 随机选择其中一种
#         GaussNoise(var_limit=(5.0, 30.0), mean=0, p=0.5),
#         ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5)
#     ], p=0.3),
#     OneOf([
#         MotionBlur(p=0.5, blur_limit=(5, 7)),  # 用随机大小的内核将运动模糊应用于输入图像，
#         MedianBlur(blur_limit=3, p=0.1),  # 中值滤波
#         Blur(blur_limit=3, p=0.1),  # 使用随即大小的内核模糊输入图像
#         GaussianBlur(blur_limit=(1, 7), sigma_limit=0, p=0.25)  # 高斯滤波
#     ], p=0.2),
#     ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
#     # RandomBrightnessContrast(p=0.2),
#     # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ToTensorV2(transpose_mask=True)
# ])
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask


class Resize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)  PIL: (w, h)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)


# Transform Data.
joint_transform = Compose([
    RandomHorizontallyFlip(),
    Resize((512, 512))
])
img_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()
