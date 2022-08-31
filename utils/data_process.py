# -*- coding: utf-8 -*-

"""
@Time : 2022/5/10
@Author : FaweksLee
@File : data_process
@Description : 
"""
import logging
import math
import os
from functools import partial
from glob import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2
import torch
from torch.utils import model_zoo
from torchvision.transforms import transforms


class ImageFolder(Dataset):
    def __init__(self, imgs_dir, labels_dir, joint_transform=None, image_transform=None, target_transform=None):
        self.imgs_dir = imgs_dir
        self.labels_dir = labels_dir
        self.joint_transform = joint_transform
        self.img_transform = image_transform
        self.target_transform = target_transform
        self.imgs_path = glob(os.path.join(self.imgs_dir, "*.png"))
        self.labels_path = glob(os.path.join(self.labels_dir, "*.png"))
        self.ids = [os.path.splitext(file)[0] for file in self.imgs_path if not file.startswith('.')]
        # assert len(self.labels_path) == len(self.imgs_path), "The numbers of labels is not compared to images"
        logging.info(r'Creating dataset with {} examples'.format(len(self.ids)))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        img = Image.open(img_path).convert('RGB')
        gt_path = self.labels_path[index]
        target = Image.open(gt_path).convert('L')
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


# 权重初始化
def weights_init(net, init_type='kaiming', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
        print('initialize network with %s type' % init_type)
        net.apply(init_func)


def load_url(url, model_dir='../logs', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return torch.load(cached_file, map_location=map_location)
    else:
        return model_zoo.load_url(url, model_dir=model_dir)

