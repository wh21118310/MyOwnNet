# -*- coding: utf-8 -*-

"""
@Time : 2022/5/18
@Author : FaweksLee
@File : train
@Description : 
"""
import copy
import os
from itertools import chain
from os.path import join, exists

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.scheduler import CosineLRScheduler
from torch import optim
from torch.backends import cudnn
from torch.cuda.amp import autocast
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DistributedSampler, DataLoader
from tqdm import tqdm
from nets.convnext import ConvNeXt_For_Seg
from utils.arguments import get_args_parser
from utils.callbacks import initial_logger, AverageMeter
from utils.data_process import weights_init, DataSetWithSupervised
from utils.get_metric import binary_accuracy, Acc, FWIoU, smooth
from utils.transform import transforms

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def tensor2img(tensor):
    image = tensor.squeeze(dim=0).cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = image[:, :, ::-1]
    image = np.float32(image) / 255
    return image


def save_hotmpas(img, epoch, idx):
    save_path = "../epochs/output_img"
    save_name = str(epoch) + "_" + str(idx) + "_cam.jpg"
    save_name = os.path.join(save_path, save_name)
    cv2.imwrite(save_name, img)


params = get_args_parser()
pretrained = False
model_path = ""

'''Loading Model'''
model_name = 'ConvNeXt'
# model = DeepLab(num_classes=2, backbone="mobilenet", pretrained=pretrained, downsample_factor=32)
model = ConvNeXt_For_Seg(3, 3)
Cuda, local_rank, distributed, device, GPU_Count = params['cuda'], params['local_rank'], params["distributed"], params[
    "device"], params["GPU_Count"]
clip_grad = params["clip_grad"]
if not pretrained:
    weights_init(model)
if model_path != "":
    if local_rank == 0:
        print('Load weights {}.'.format(model_path))
    # ------------------------------------------------------#
    #   根据预训练权重的Key和模型的Key进行加载
    # ------------------------------------------------------#
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, mpa_location=params["device"])
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
    if local_rank == 0:
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
if Cuda:
    if distributed:
        model = model.cuda(local_rank)
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        model = DataParallel(model)
        cudnn.benchmark = True  # 当模型架构保持不变以及输入大小保持不变时可使用
        model = model.cuda()

'''Loading Datasets'''
batch_size, num_workers = params['batch_size'], params['num_workers']
data_dir = r"dataset"
train_imgs_dir, val_imgs_dir, test_imgs_dir = join(data_dir, "train/img"), join(data_dir, "valid/img"), join(data_dir,
                                                                                                             "test/img")
train_labels_dir, val_labels_dir, test_labels_dir = join(data_dir, "train/gt"), join(data_dir, "valid/gt"), join(
    data_dir, "test/gt")
train_data = DataSetWithSupervised(train_imgs_dir, train_labels_dir, transforms)
val_data = DataSetWithSupervised(val_imgs_dir, val_labels_dir, transforms)
test_data = DataSetWithSupervised(test_imgs_dir, test_labels_dir, transforms)
train_data_size, valid_data_size, test_data_size = train_data.__len__(), val_data.__len__(), test_data.__len__()
if distributed:
    train_sampler = DistributedSampler(train_data, shuffle=True)
    val_sampler = DistributedSampler(val_data, shuffle=False)
    test_sampler = DistributedSampler(test_data, shuffle=True)
    batch_size = batch_size // GPU_Count
    shuffle = False
else:
    # train_sampler = RandomSampler(train_data)
    # val_sampler = SequentialSampler(val_data)
    # test_sampler = SequentialSampler(test_data)
    train_sampler, val_sampler, test_sampler = None, None, None
    shuffle = True
train_loader = DataLoader(train_data, shuffle=shuffle, batch_size=batch_size,
                          num_workers=num_workers, sampler=train_sampler, pin_memory=True)
val_loader = DataLoader(val_data, shuffle=shuffle, batch_size=batch_size,
                        num_workers=num_workers, sampler=val_sampler, pin_memory=True)
test_loader = DataLoader(test_data, shuffle=shuffle, batch_size=batch_size,
                         num_workers=num_workers, sampler=test_sampler, pin_memory=True)
trainLoader_size, valLoader_size, testLoader_size = len(train_loader), len(val_loader), len(test_loader)
'''For Epoch'''
Init_Epoch = params["Init_Epoch"]
Total_epoch = params["Total_Epoch"]
save_epoch = 10  # 多少个epoch保存一次权值
'''Loading Optimizer'''
Init_lr_fit, momentum, weight_decay, Min_lr = params['Init_lr_fit'], params['momentum'], params['weight_decay'], params[
    'Min_lr']
optimizer_type, lr_decay_type = params['Optimizer'], params['lr_decay']
optimizer = {
    'adam': optim.Adam(chain(model.parameters()), Init_lr_fit, betas=(momentum, 0.999),
                       weight_decay=weight_decay),
    'sgd': optim.SGD(chain(model.parameters()), Init_lr_fit, momentum=momentum, nesterov=True,
                     weight_decay=weight_decay),
}[optimizer_type]
''''Loading Scheduler'''
scheduler = {
    'cos': CosineLRScheduler(optimizer, t_initial=50, t_mul=1.0, lr_min=Min_lr,
                             decay_rate=weight_decay, warmup_t=0, warmup_lr_init=Init_lr_fit),
    # lr = 0.05 if epoch < 30; lr= 0.005 if 30 <= epoch < 60; lr = 0.0005 if 60 <= epoch < 90
    'steplr': StepLR(optimizer, step_size=int(Total_epoch / 3), gamma=0.1)
}[lr_decay_type]

'''Loading Scaler'''
scaler = params["scaler"]

'''Loading criterion'''
criterion = params["criterion"]

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

best_iou, best_epoch, best_mpa, epoch_start, best_mode = 0, 0, 0, Init_Epoch, copy.deepcopy(model)

save_inter, min_inter = 100, 0  # 用来存模型

# support for the restart from breakpoint
if Resume and exists(best_ckpt):
    checkpoint = torch.load(best_ckpt)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch_start = checkpoint['epoch']
    scheduler.load_state_dict(checkpoint['scheduler'])

# training loss curve
plot = True

logger.info('Total Epoch:{} Training num:{}  Validation num:{}'.format(
    Total_epoch, train_data_size, valid_data_size))
for epoch in range(epoch_start, Total_epoch):
    model.train()
    train_main_loss = AverageMeter()
    train_bar = tqdm(train_loader)
    for batch_idx, (image, label) in enumerate(train_bar, start=1):
        optimizer.zero_grad(set_to_none=True)
        if Cuda:
            image = image.cuda(local_rank)
            image = image.to(dtype=torch.float32, non_blocking=True)
            label = label.cuda(local_rank)
            label = label.to(dtype=torch.float32, non_blocking=True)
        if scaler is None:
            outputs = model(image)
            loss = criterion(outputs, label)
            loss.backward()
            if clip_grad:
                clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
            optimizer.step()
        else:
            with autocast():
                outputs = model(image)
                loss = criterion(outputs, label)
            scaler.scale(loss).backward()
            if clip_grad:
                clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
            scaler.step(optimizer)
            scaler.update()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        scheduler.step(epoch + batch_idx / trainLoader_size)  # called after every batch update
        train_main_loss.update(loss.cpu().detach().numpy())
        train_bar.set_description(desc='[train] epoch:{} iter:{}/{} lr:{:.4f} loss:{:.4f}'.format(
            epoch, batch_idx, trainLoader_size, optimizer.param_groups[-1]['lr'], train_main_loss.average()))
        if batch_idx == trainLoader_size:
            logger.info('[train] epoch:{} iter:{}/{} lr:{:.4f} loss:{:.4f}'.format(
                epoch, batch_idx, trainLoader_size, optimizer.param_groups[-1]['lr'], train_main_loss.average()))

    model.eval()
    val_bar = tqdm(val_loader)
    val_loss = AverageMeter()
    acc_meter = AverageMeter()
    fwIoU_meter = AverageMeter()
    mpa_meter = AverageMeter()
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(val_bar, start=1):
            if Cuda:
                image = image.cuda(local_rank)
                label = label.cuda(local_rank)
                image = image.to(dtype=torch.float32, non_blocking=True)
                label = label.to(dtype=torch.float32, non_blocking=True)
            outputs = model(image)
            loss = criterion(outputs, label)
            val_loss.update(loss.cpu().detach().numpy())
            val_bar.set_description(desc='[val] epoch:{} iter:{}/{} loss:{:.4f}'.format(
                epoch, batch_idx, valLoader_size, val_loss.average()))
            if batch_idx == valLoader_size:
                logger.info('[val] epoch:{} iter:{}/{} loss:{:.4f}'.format(
                    epoch, batch_idx+1, valLoader_size, val_loss.average()))
            '''以下部分由于数据集的图片是二分类图像，故采用以下方式处理'''
            # outputs = torch.sigmoid(outputs)
            outputs = torch.where(outputs > 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))
            outputs = outputs.cpu().detach().numpy()
            for (outputs, target) in zip(outputs, label):
                acc, valid_sum = binary_accuracy(outputs, target)
                mpa = Acc(outputs.squeeze(), target.cpu().squeeze(), ignore_zero=False)
                fwiou = FWIoU(outputs.squeeze(), target.cpu().squeeze(), ignore_zero=False)
                acc_meter.update(acc)
                mpa_meter.update(mpa)
                fwIoU_meter.update(fwiou)
    # save loss & lr
    train_loss_total_epochs.append(train_main_loss.avg)
    valid_loss_total_epochs.append(val_loss.average())
    epoch_lr.append(optimizer.param_groups[-1]['lr'])
    # save Model
    if fwIoU_meter.average() > best_iou or epoch == 0 or (epoch % save_inter == 0 and epoch > min_inter):
        state = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        filename = join(save_ckpt_dir, 'ckpt-epoch{}_fwiou{:.2f}.pth'.format(epoch, fwIoU_meter.avg * 100))
        torch.save(state, filename, _use_new_zipfile_serialization=False)
        if fwIoU_meter.average() > best_iou:
            best_filename = join(save_ckpt_dir, 'best_model.pth')
            torch.save(state, best_filename, _use_new_zipfile_serialization=False)
            best_iou = fwIoU_meter.average()
            best_mode = copy.deepcopy(model)
            best_epoch = epoch
            logger.info('[save] Best Model saved at epoch:{}'.format(epoch))
    if mpa_meter.average() > best_mpa:
        best_mpa = mpa_meter.average()
    # 显示loss
    print("best_epoch:{}, nowIoU: {:.4f}, bestIoU:{:.4f}, now_mPA:{:.4f}, best_mPA:{:.4f}"
          .format(best_epoch, fwIoU_meter.average() * 100, best_iou * 100, mpa_meter.average() * 100, best_mpa * 100))
if plot:
    x = [i for i in range(Total_epoch)]
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x, smooth(train_loss_total_epochs, 0.6), label='train loss')
    ax.plot(x, smooth(valid_loss_total_epochs, 0.6), label='val loss')
    ax.set_xlabel('Epoch', fontsize=15)
    ax.set_ylabel('CrossEntropy', fontsize=15)
    ax.set_title('train curve', fontsize=15)
    ax.grid(True)
    plt.legend(loc='upper right', fontsize=15)
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x, epoch_lr, label='Learning Rate')
    ax.set_xlabel('Epoch', fontsize=15)
    ax.set_ylabel('Learning Rate', fontsize=15)
    ax.set_title('lr curve', fontsize=15)
    ax.grid(True)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()
