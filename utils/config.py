# -*- coding: utf-8 -*-

"""
    @Time : 2022/8/30 12:32
    @Author : FaweksLee
    @Email : 121106010719@njust.edu.cn
    @File : config
    @Description : 
"""
from os.path import join

backbone_path = './params/resnet/resnet50.pth'
data_dir = r"dataset/MarineFarm_80"
train_imgs_dir, val_imgs_dir = join(data_dir, "train/images"), join(data_dir, "val/images")
train_labels_dir, val_labels_dir = join(data_dir, "train/gt"), join(data_dir, "val/gt")
