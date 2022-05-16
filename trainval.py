# -*- coding: utf-8 -*-

"""
@Time : 2022/5/10
@Author : FaweksLee
@File : train
@Description : Training and Validation
"""
import os, torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset,RandomSampler


if __name__ == '__main__':
    Cuda = True  # 没有GPU可设定为False
    '''
    distributed     用于指定是否使用单机多卡分布式运行
                    终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
                    Windows系统下默认使用DP模式调用所有显卡，不支持DDP。
    DP模式：
        设置            distributed = False
        在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python train.py
    DDP模式：
        设置            distributed = True
        在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    '''
    distributed = False
    sync_bn = False            # 是否使用sync_bn,DDP模式下多卡可用
    fp16 = False               # 是否使用混合精度训练，以减少一半显存，需要pytorch1.7.1以上
    num_classes = 21           # 数据集包含的类别数，包括背景（作为一类），VOC2007数据集有20类
    backbone = "mobilenet"     # 复杂网络会采用某种网络作为backbone，默认应为None
    network = "deeplabv3plus"  # 网络名称，根据名称载入网络


