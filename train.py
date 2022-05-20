# -*- coding: utf-8 -*-

"""
@Time : 2022/5/18
@Author : FaweksLee
@File : train
@Description : 
"""
import os
import random
from itertools import chain

import cv2
import numpy as np
import torch
from timm.scheduler import CosineLRScheduler
from torch import optim
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DistributedSampler, DataLoader

from nets.deeplabv3Plus import DeepLab
from utils.arguments import get_args_parser
from utils.data_process import weights_init, DataSetWithSupervised

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def tensor2img(tensor):
    image = tensor.squeeze(dim=0).cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = image[:, :, ::-1]
    image = np.float32(image) / 255
    return image


def save_hotmaps(img, epoch, idx):
    save_path = "../epochs/output_img"
    save_name = str(epoch) + "_" + str(idx) + "_cam.jpg"
    save_name = os.path.join(save_path, save_name)
    cv2.imwrite(save_name, img)


params = get_args_parser()
data_path = "dataset"
pretrained = False
model_path = ""
'''Loading Model'''
model = DeepLab(num_classes=2, backbone="mobilenet", pretrained=pretrained, downsample_factor=32)
if not pretrained:
    weights_init(model)
if model_path != "":
    if params["local_rank"] == 0:
        print('Load weights {}.'.format(model_path))
    # ------------------------------------------------------#
    #   根据预训练权重的Key和模型的Key进行加载
    # ------------------------------------------------------#
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=params["device"])
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    # ------------------------------------------------------#
    #   显示没有匹配上的Key
    # ------------------------------------------------------#
    if params["local_rank"] == 0:
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
# # ----------------------------#
# #   多卡同步Bn
# # ----------------------------#
# if params["sync_bn"] and params["ngpus_per_node"] > 1 and params["distributed"]:
#     model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
# elif params["sync_bn"]:
#     print("Sync_bn is not support in one gpu or not distributed.")
if params["cuda"]:
    if params["distributed"]:
        model = model.cuda(params["local_rank"])
        model = DistributedDataParallel(model, device_ids=[params["local_rank"]], find_unused_parameters=True)
    else:
        model = DataParallel(model)
        cudnn.benchmark = True
        model = model.cuda()
'''Loading Datasets'''
TrainValImgs = "dataset/JPEGImages"
TrainValGT = "dataset/SegmentationClass"
train_dataset = DataSetWithSupervised(TrainValImgs, TrainValGT)
val_dataset = DataSetWithSupervised(TrainValImgs, TrainValGT)
if params['distributed']:
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    batch_size = params['batch_size'] // params['ngpus_per_node']
    shuffle = False
else:
    train_sampler = None
    val_sampler = None
    shuffle = True
train_loader = DataLoader(train_dataset, shuffle=shuffle, batch_size=params['batch_size'],
                          num_workers=params['num_workers'], sampler=train_sampler)
val_loader = DataLoader(val_dataset, shuffle=shuffle, batch_size=params['batch_size'],
                        num_workers=params["num_workers"], sampler=val_sampler)
'''Loading Optimizer'''
optimizer = {
    'adam': optim.Adam(chain(model.parameters()), params['Init_lr_fit'], betas=(params["momentum"], 0.999), weight_decay=params["weight_decay"]),
    'sgd': optim.SGD(chain(model.parameters()), params['Init_lr_fit'], momentum=params['momentum'], nesterov=True, weight_decay=params["weight_decay"])
}[params['optimizer_type']]
''''Loading Scheduler'''
scheduler = {
    'cos': CosineLRScheduler(optimizer, t_initial=50, t_mul=1.0, lr_min=params['Min_lr'],
                             decay_rate=params["weight_decay"], warmup_t=0, warmup_lr_init=params['Init_lr_fit']),
    # lr = 0.05 if epoch < 30; lr= 0.005 if 30 <= epoch < 60; lr = 0.0005 if 60 <= epoch < 90
    'steplr': StepLR(optimizer, step_size=int(params['Total_Epoch']/3), gamma=0.1)
}[params['lr_decay_type']]
