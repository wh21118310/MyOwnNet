# -*- coding: utf-8 -*-

"""
@Time : 2022/5/18
@Author : FaweksLee
@File : train
@Description : 
"""
import copy
import os
import random
from itertools import chain
from os.path import join, exists

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
from utils.callbacks import initial_logger
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
pretrained = False
model_path = ""
'''Loading Model'''
model_name = 'deeplabv3p'
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
if params["cuda"]:
    if params["distributed"]:
        model = model.cuda(params["local_rank"])
        model = DistributedDataParallel(model, device_ids=[params["local_rank"]], find_unused_parameters=True)
    else:
        model = DataParallel(model)
        cudnn.benchmark = True
        model = model.cuda()
'''Loading Datasets'''
data_dir = r"dataset"
train_imgs_dir, val_imgs_dir, test_imgs_dir = join(data_dir, "train/img"), join(data_dir, "valid/img"), join(data_dir,
                                                                                                             "test/img")
train_labels_dir, val_labels_dir, test_labels_dir = join(data_dir, "train/gt/"), join(data_dir, "valid/gt/"), join(
    data_dir, "test/gt/")
train_data = DataSetWithSupervised(train_imgs_dir, train_labels_dir)
val_data = DataSetWithSupervised(val_imgs_dir, val_labels_dir)
test_data = DataSetWithSupervised(test_imgs_dir, test_labels_dir)
train_data_size, valid_data_size, test_data_size = train_data.__len__(), val_data.__len__(), test_data.__len__()
if params['distributed']:
    train_sampler = DistributedSampler(train_data, shuffle=True)
    val_sampler = DistributedSampler(val_data, shuffle=False)
    test_sampler = DistributedSampler(test_data, shuffle=True)
    batch_size = params['batch_size'] // params['ngpus_per_node']
    shuffle = False
else:
    train_sampler, val_sampler, test_sampler = None, None, None
    shuffle = True
train_loader = DataLoader(train_data, shuffle=shuffle, batch_size=params['batch_size'],
                          num_workers=params['num_workers'], sampler=train_sampler)
val_loader = DataLoader(val_data, shuffle=shuffle, batch_size=params['batch_size'],
                        num_workers=params["num_workers"], sampler=val_sampler)
test_loader = DataLoader(test_data, shuffle=shuffle, batch_size=params['batch_size'],
                         num_workers=params["num_workers"], sampler=test_sampler)
'''Loading Optimizer'''
optimizer = {
    'adam': optim.Adam(chain(model.parameters()), params['Init_lr_fit'], betas=(params["momentum"], 0.999),
                       weight_decay=params["weight_decay"]),
    'sgd': optim.SGD(chain(model.parameters()), params['Init_lr_fit'], momentum=params['momentum'], nesterov=True,
                     weight_decay=params["weight_decay"]),
}[params['optimizer_type']]
''''Loading Scheduler'''
scheduler = {
    'cos': CosineLRScheduler(optimizer, t_initial=50, t_mul=1.0, lr_min=params['Min_lr'],
                             decay_rate=params["weight_decay"], warmup_t=0, warmup_lr_init=params['Init_lr_fit']),
    # lr = 0.05 if epoch < 30; lr= 0.005 if 30 <= epoch < 60; lr = 0.0005 if 60 <= epoch < 90
    'steplr': StepLR(optimizer, step_size=int(params['Total_Epoch'] / 3), gamma=0.1)
}[params['lr_decay_type']]
'''Loading Scaler'''
scaler = params["scaler"]
'''Loading criterion'''
criterion = params["criterion"]
'''For Epoch'''
Init_Epoch = params["Init_Epoch"]
Total_Epch = params["Total_Epoch"]
save_epoch = 10  # 多少个epoch保存一次权值
'''Save Path'''
save_dir = './outputs'  # 权值与日志文件保存的文件夹
save_ckpt_dir, save_log_dir = join(save_dir, 'ckpt'), join(save_dir, 'log')
best_ckpt = join(save_ckpt_dir, 'best_model.pth')
if not exists(save_ckpt_dir):
    os.makedirs(save_ckpt_dir)
if not exists(save_log_dir):
    os.makedirs(save_log_dir)
'''Logger'''
logger = initial_logger(join(save_log_dir, model_name + '.log'))
'''Main Iteration'''
train_loss_total_epochs, valid_loss_total_epochs, epoch_lr = list(), list(), list()
Resume = False  # used for discriminate the status from the breakpoint or start
best_iou, best_epoch, best_mpa, epoch_start, best_mode = 0.5, 0, 0.5, Init_Epoch, copy.deepcopy(model)
if Resume and exists(best_ckpt):
    checkpoint = torch.load(best_ckpt)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    scheduler.load_state_dict(checkpoint['scheduler'])
logger.info('Total Epoch:{} Training num:{}  Validation num:{}'.format(
        Total_Epch, train_data_size, valid_data_size))
for epoch in range(epoch_start, Total_Epch):

