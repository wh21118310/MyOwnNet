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
from torch.utils.data.dataset import Dataset
import cv2
import torch
from torch.utils import model_zoo

# # 载入预训练权重
# def load_url(url, model_dir='./pretrained', map_location=None):
#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)
#     filename = url.split('/')[-1]
#     cached_file = os.path.join(model_dir, filename)
#     if os.path.exists(cached_file):
#         return torch.load(cached_file, map_location=map_location)
#     else:
#         return model_zoo.load_url(url, model_dir=model_dir)
#
#
# # 网络初始化
# def weights_init(net, init_type='normal', init_gain=0.02):
#     def init_func(m):
#         classname = m.__class__.__name__
#         if hasattr(m, 'weight') and classname.find('Conv') != -1:
#             if init_type == 'normal':
#                 torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
#             elif init_type == 'xavier':
#                 torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
#             elif init_type == 'kaiming':
#                 torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#             elif init_type == 'orthogonal':
#                 torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
#             else:
#                 raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
#         elif classname.find('BatchNorm2d') != -1:
#             torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#             torch.nn.init.constant_(m.bias.data, 0.0)
#
#     print('initialize network with %s type' % init_type)
#     net.apply(init_func)


class DataSetWithSupervised(Dataset):
    def __init__(self, imgs_dir, labels_dir, tfs=None):
        self.imgs_dir = imgs_dir
        self.labels_dir = labels_dir
        self.transform = tfs
        try:
            self.imgs_path = glob(os.path.join(self.imgs_dir, "*.png"))
        except Exception:
            self.imgs_path = glob(os.path.join(self.imgs_dir, "*.jpg"))
        try:
            self.labels_path = glob(os.path.join(self.labels_dir, ".png"))
        except Exception:
            self.labels_path = glob(os.path.join(self.labels_dir, "*.jpg"))
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        self.useOneDim = False

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # get the idx-th image
        image_path = self.imgs_path[idx]
        # get the label correspond to the image
        label_path = self.labels_path[idx]
        # read the image and label
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # if want the result to get the 1d image
        if self.useOneDim:
            label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)
            label = self.transform(image=label)
        return image, label / 255


class DataSetWithNosupervised(Dataset):
    def __init__(self, imgs_dir, tfs=None):
        self.imgs_dir = imgs_dir
        self.transform = tfs
        try:
            self.imgs_path = glob(os.path.join(self.imgs_dir, "*.png"))
        except Exception:
            self.imgs_path = glob(os.path.join(self.imgs_dir, "*.jpg"))
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # get the idx-th image
        image_path = self.imgs_path[idx]
        # read the image and label
        image = cv2.imread(image_path)
        # if want the result to get the 1d image
        if self.transform is not None:
            image = self.transform(image=image)
        return image


# 权重初始化
def weights_init(net, init_type='normal', init_gain=0.02):
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


# def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.1, warmup_lr_ratio=0.1,
#                      no_aug_iter_ratio=0.3, step_num=10):
#     def warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
#         if iters <= warmup_total_iters:
#             # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
#             lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
#         elif iters >= total_iters - no_aug_iter:
#             lr = min_lr
#         else:
#             lr = min_lr + 0.5 * (lr - min_lr) * (
#                     1.0 + math.cos(math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
#             )
#         return lr
#
#     def step_lr(lr, decay_rate, step_size, iters):
#         if step_size < 1:
#             raise ValueError("step_size must above 1.")
#         n = iters // step_size
#         out_lr = lr * decay_rate ** n
#         return out_lr
#
#     if lr_decay_type == "cos":
#         warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
#         warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
#         no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
#         func = partial(warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
#     else:
#         decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
#         step_size = total_iters / step_num
#         func = partial(step_lr, lr, decay_rate, step_size)
#
#     return func
#
#
# def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
#     lr = lr_scheduler_func(epoch)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
def load_url(url, model_dir='../logs', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return torch.load(cached_file, map_location=map_location)
    else:
        return model_zoo.load_url(url, model_dir=model_dir)