# -*- coding: utf-8 -*-

"""
@Time : 2022/5/18
@Author : FaweksLee
@File : train
@Description :
"""
import copy
import os
from os.path import join, exists

import torch
from timm.utils import ModelEma, get_state_dict
from torch.backends import cudnn

from torch.cuda.amp import autocast

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DistributedSampler, DataLoader
from tqdm import tqdm

# from nets.PFNet import PFNet, PFNet_withPVT
from nets.PFNet_ASPP import PFNet
from nets.SINet import SearchIdentificationNet as SInet
from nets.backbone.PSPNet import Pspnet
from nets.backbone.Swin_transformer import SwinNet
from nets.backbone.convnext import ConvNeXt_Seg
from utils.arguments import get_scaler, get_opt_and_scheduler, get_criterion, check_path, seed_torch, \
    distributedTraining, model_load, get_mixup, ModelToCuda, initial_logger, AverageMeter, draw
from utils.data_process import weights_init, MarineFarmData
from utils.get_metric import binary_accuracy, Acc, FWIoU
from utils.transform import transforms
from utils.arguments import structure_loss as criterion

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# params = get_args_parser()

'''Loading Model'''
seed_torch(seed=2022)
model_name = 'PFNet_convnext_AS_80'
model = PFNet(bk="convnext_base")
# model_name = 'PFNet_swinT_AS_80'
# model = PFNet(bk='swinT_base')
# model_name = 'SINet_convnext_base_80'
# model = SInet(bk='convnext_base')
# model_name = 'SINet_swinT_base_80'
# model = SInet(bk="swinT_base")
# model_name = 'SINet_res2net50_80'
# model = SInet(bk='res2net50')
# model_name = 'PFNet_PVT_large_80'
# model = PFNet_withPVT(bk="large", img_size=512)
# model_name = 'PSPNet_1b'
# model = Pspnet(num_classes=1)
gpu_id = "0"

Cuda = True
distributed = False
pretrained = True
model_path = ""
# local_rank, device, GPU_Count = distributedTraining(distributed)
if gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("Let's use GPU 0")
elif gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print("Let's use GPU 1")
# cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
local_rank = 0
if not pretrained:
    weights_init(model)
model = model.cuda()

'''Loading Datasets'''
batch_size = 4
data_dir = r"dataset/MarineFarm_80"
train_imgs_dir, val_imgs_dir, test_imgs_dir = join(data_dir, "train/images"), join(data_dir, "val/images"), \
                                              join(data_dir, "test/images")
train_labels_dir, val_labels_dir, test_labels_dir = join(data_dir, "train/gt"), join(data_dir, "val/gt"), \
                                                    join(data_dir, "test/gt")
train_data = MarineFarmData(train_imgs_dir, train_labels_dir, transforms)
val_data = MarineFarmData(val_imgs_dir, val_labels_dir, transforms)
test_data = MarineFarmData(test_imgs_dir, test_labels_dir, transforms)
train_data_size, valid_data_size, test_data_size = train_data.__len__(), val_data.__len__(), test_data.__len__()
if distributed:
    train_sampler = DistributedSampler(train_data, shuffle=True)
    val_sampler = DistributedSampler(val_data, shuffle=False)
    test_sampler = DistributedSampler(test_data, shuffle=True)
    batch_size = batch_size // 2
    shuffle = False
else:
    train_sampler, val_sampler, test_sampler = None, None, None
    shuffle = True
# ------------------------------------------------------------------#
# num_workers用于设置是否使用多线程读取数据，1代表关闭多线程。开启后会加快数据读取速度，但是会占用更多内存。
# Windows只可设定为0
# ------------------------------------------------------------------#
num_workers = 0
train_loader = DataLoader(train_data, shuffle=shuffle, batch_size=batch_size,
                          num_workers=num_workers, sampler=train_sampler, pin_memory=True)
val_loader = DataLoader(val_data, shuffle=shuffle, batch_size=batch_size,
                        num_workers=num_workers, sampler=val_sampler, pin_memory=True)
test_loader = DataLoader(test_data, shuffle=shuffle, batch_size=batch_size,
                         num_workers=num_workers, sampler=test_sampler, pin_memory=True)
trainLoader_size, valLoader_size, testLoader_size = len(train_loader), len(val_loader), len(test_loader)

'''MixUp Strategy'''
smooth = 0.
# mixup, cutmix, cutmix_minmax, mixup_prob, mixup_switch_prob, mixup_mode = 0.8, 1.0, None, 1.0, 0.5, "batch"
# mixup_fn = get_mixup(mixup, cutmix, cutmix_minmax, mixup_prob, mixup_switch_prob, mixup_mode, smooth, num_classes=2)

'''Loading criterion'''
criterion_name = "bcew"
# criterion, mixup_fn = get_criterion(loss_name=criterion_name, mixup=None, smoothing=smooth, is_gpu=Cuda)
'''EMA Strategy'''
model_ema = False
model_ema_decay = 0.999
model_ema_force_cpu = False
# model_ema_eval = False
ema = None
if model_ema:
    ema = ModelEma(model, decay=model_ema_decay, device='cpu' if model_ema_force_cpu else '', resume='')
    print("Using EMA with decay = %.8f" % model_ema_decay)

'''Loading Optimizer and Scheduler'''
optimizer_type = "sgd"
momentum = 0.9
lr_decay = 'cos'
optimizer, scheduler = get_opt_and_scheduler(model=model, optimizer_type=optimizer_type, lr_decay_type=lr_decay,
                                             momentum=momentum)
'''Loading Scaler'''
fp16 = False
scaler = get_scaler(fp16)

'''Save Path'''
nowPath = os.getcwd()
save_dir = join(nowPath, 'out', model_name)  # 权值与日志文件保存的文件夹
check_path(save_dir)
save_ckpt_dir, save_log_dir = join(save_dir, 'ckpt'), join(save_dir, 'log')
best_ckpt = join(save_ckpt_dir, 'best_model.pth')
check_path(save_ckpt_dir)
check_path(save_log_dir)
'''For Epoch & Restart'''
epoch_start, Total_epoch = 0, 100
Resume = False  # used for discriminate the status from the breakpoint or start
# support for the restart from breakpoint
if Resume and exists(best_ckpt):
    checkpoint = torch.load(best_ckpt)
    model.load_state_dict(checkpoint['backbone'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch_start = checkpoint['epoch']
    scheduler.load_state_dict(checkpoint['scheduler'])

'''Logger'''
logger = initial_logger(join(save_log_dir, model_name + '.log'))
'''Main Iteration'''
train_loss_total_epochs, valid_loss_total_epochs, epoch_lr = list(), list(), list()
epoch_iou = list()  # save index
'''Train & val'''
clip_grad = False
best_iou, best_epoch, best_mpa = .5, 0, .5
last_index, save_iter = 0., 0
save_inter, min_inter = 10, 10  # 用来存模型
last_file = None
logger.info('Total Epoch:{} Training num:{}  Validation num:{}'.format(
    Total_epoch, train_data_size, valid_data_size))
for epoch in range(epoch_start, Total_epoch):
    model.train()
    train_main_loss = AverageMeter()
    train_bar = tqdm(train_loader)
    for batch_idx, (image, label) in enumerate(train_bar, start=1):
        optimizer.zero_grad(set_to_none=True)
        if Cuda:
            # images = images.cuda(local_rank)
            image = image.to(device=device, dtype=torch.float32, non_blocking=True)
            # label = label.cuda(local_rank)
            label = label.to(device=device, dtype=torch.float32, non_blocking=True)
        # if mixup_fn is not None:
        #     image, label = mixup_fn(image, label)
        if scaler is None:
            outputs = model(image)
            loss = criterion(outputs, label.unsqueeze(1))
            loss.backward()
            if clip_grad:
                clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
            optimizer.step()
        else:
            with autocast():
                outputs = model(image)
                loss = criterion(outputs, label.unsqueeze(1))
                # loss = criterion(outputs[0].squeeze(1), label) + criterion(outputs[1].squeeze(1), label) +
                # criterion(outputs[2].squeeze(1), label) + criterion(outputs[3].squeeze(1), label)
            scaler.scale(loss).backward()
            if clip_grad:
                clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
            scaler.step(optimizer)
            scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model)
        scheduler.step(epoch + batch_idx / trainLoader_size)  # called after every batch update
        # scheduler.step()  # when use stepLR
        train_main_loss.update(loss.cpu().detach().numpy())
        train_bar.set_description(desc='[train] epoch:{} iter:{}/{} lr:{:.4f} loss:{:.4f}'.format(
            epoch, batch_idx, trainLoader_size, optimizer.param_groups[-1]['lr'], train_main_loss.average()))
        if batch_idx == trainLoader_size:
            logger.info('[train] epoch:{} iter:{}/{} lr:{:.4f} loss:{:.4f}'.format(
                epoch, batch_idx, trainLoader_size, optimizer.param_groups[-1]['lr'], train_main_loss.average()))
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    model.eval()
    val_bar = tqdm(val_loader)
    val_loss = AverageMeter()
    acc_meter = AverageMeter()
    fwIoU_meter = AverageMeter()
    mpa_meter = AverageMeter()
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(val_bar, start=1):
            if Cuda:
                image = image.to(device=device, dtype=torch.float32, non_blocking=True)
                label = label.to(device=device, dtype=torch.float32, non_blocking=True)
            # if mixup_fn is not None:
            #     image, label = mixup_fn(image, label)
            outputs = model(image)
            loss = criterion(outputs, label.unsqueeze(1))
            val_loss.update(loss.cpu().detach().numpy())
            val_bar.set_description(desc='[val] epoch:{} iter:{}/{} loss:{:.4f}'.format(
                epoch, batch_idx, valLoader_size, val_loss.average()))
            if batch_idx == valLoader_size:
                logger.info('[val] epoch:{} iter:{}/{} loss:{:.4f}'.format(
                    epoch, batch_idx, valLoader_size, val_loss.average()))
            '''以下部分由于数据集的图片是二分类图像，故采用以下方式处理'''
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
    save_iter += 1
    if fwIoU_meter.average() > best_iou or epoch > min_inter:
        state = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        if ema is not None:
            state['model_ema'] = get_state_dict(ema)
        if last_index == 0.:
            filename = join(save_ckpt_dir, 'ckpt-epoch{}_fwiou{:.3f}.pth'.format(epoch, fwIoU_meter.average() * 100))
            torch.save(state, filename, _use_new_zipfile_serialization=False)
            last_index = fwIoU_meter.average()
            # save_iter = 0
            last_file = filename
        elif fwIoU_meter.average() - last_index >= 0:
            filename = join(save_ckpt_dir, 'ckpt-epoch{}_fwiou{:.3f}.pth'.format(epoch, fwIoU_meter.average() * 100))
            torch.save(state, filename, _use_new_zipfile_serialization=False)
            if exists(last_file):
                os.remove(last_file)
            last_file = filename
            last_index = fwIoU_meter.average()
            # save_iter = 0
        if fwIoU_meter.average() > best_iou:
            best_filename = join(save_ckpt_dir, 'best_model.pth')
            if exists(best_filename):  # del old
                os.remove(best_filename)
            torch.save(state, best_filename, _use_new_zipfile_serialization=False)
            best_iou = fwIoU_meter.average()
            best_mode = copy.deepcopy(model)
            best_epoch = epoch
            logger.info('[save] Best Model saved at epoch:{}, fwIou:{}'.format(best_epoch, fwIoU_meter.average()))
    if mpa_meter.average() > best_mpa:
        best_mpa = mpa_meter.average()
    epoch_iou.append(fwIoU_meter.average())
    # 显示loss
    print("best_epoch:{}, nowIoU: {:.4f}, bestIoU:{:.4f}, now_mPA:{:.4f}, best_mPA:{:.4f}\n"
          .format(best_epoch, fwIoU_meter.average() * 100, best_iou * 100, mpa_meter.average() * 100,
                  best_mpa * 100))

plot = True
if plot:
    draw(Total_epoch, train_loss_total_epochs, valid_loss_total_epochs, epoch_lr, epoch_iou, save_log_dir)
