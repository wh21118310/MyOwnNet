# -*- coding: utf-8 -*-

"""
@Time : 2022/5/16
@Author : FaweksLee
@File : test
@Description : 
"""
import os

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device_num = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
print(device_num)
print(os.environ['LOCAL_RANK'])