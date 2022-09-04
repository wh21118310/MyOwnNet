# -*- coding: utf-8 -*-

"""
@Time : 2022/5/15
@Author : FaweksLee
@File : arguments
@Description : The arguments
"""
import os
import random
from os.path import join

import cv2
import numpy as np
import torch
from pytorch_toolbelt.losses import JointLoss
from segmentation_models_pytorch.losses import *
from segmentation_models_pytorch.utils.losses import CrossEntropyLoss
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.scheduler.scheduler import Scheduler
from torch import optim
from torch.nn import BCELoss, MSELoss, BCEWithLogitsLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR
from timm.scheduler import CosineLRScheduler, StepLRScheduler, PlateauLRScheduler


# set seeds

def seed_torch(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.deterministic = True


def tensor2img(tensor):
    image = tensor.squeeze(dim=0).cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = image[:, :, ::-1]
    image = np.float32(image) / 255
    return image


def save_hotmpas(img, epoch, idx, save_path):
    save_path = join(save_path, "hotmap")
    save_name = str(epoch) + "_" + str(idx) + "_cam.jpg"
    save_name = join(save_path, save_name)
    cv2.imwrite(save_name, img)


def check_path(Path):
    if not os.path.exists(Path):
        print("There are no path:{}, make it now!".format(Path))
        os.makedirs(Path, exist_ok=True)


'''
1. CUDA、Parallel Setting
'''


def get_sync(distributed: bool) -> bool:
    # ---------------------------------------------------------------------#
    #   sync_bn     whether to use sync_bn in DDP multiGPU mode
    # ---------------------------------------------------------------------#
    if distributed:
        sync_bn = True
    else:
        sync_bn = False
    return sync_bn


# def model_load(model, model_path, local_rank, device):
#     if model_path != "":
#         print('Load weights {}\n'.format(model_path))
#         # ------------------------------------------------------#
#         #   根据预训练权重的Key和模型的Key进行加载
#         # ------------------------------------------------------#
#         model_dict = model.state_dict()
#         pretrained_dict = torch.load(model_path, mpa_location=device)
#         load_key, no_load_key, temp_dict = [], [], {}
#         for k, v in pretrained_dict.items():
#             if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
#                 temp_dict[k] = v
#                 load_key.append(k)
#             else:
#                 no_load_key.append(k)
#         model_dict.update(temp_dict)
#         model.load_state_dict(model_dict)
#         # ------------------------------------------------------#
#         #   显示没有匹配上的Key
#         # ------------------------------------------------------#
#         if local_rank == 0:
#             print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
#             print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
#             print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
#     return model
def check_UpOrLower(varName: str):
    if not varName.islower():
        return varName.lower()
    else:
        return varName


def get_optimizer(model, args):
    optimizer_type = args['optimizer']
    optimizer_type = check_UpOrLower(optimizer_type)
    weight_decay = {"sgd": 1e-4, "adam": 0}[optimizer_type]
    weight_decay = weight_decay if weight_decay > args['weight_decay'] else args['weight_decay']
    Init_lr = {"sgd": 7e-3, "adam": 5e-4}[optimizer_type]  # Set as recommand
    Init_lr = Init_lr if Init_lr > args['lr'] else args['lr']
    momentum = 0.9
    if args['momentum'] is not None:
        momentum = args['momentum']
    # Min_lr = max_lr *0.01
    optimizer = {
        'adam': optim.Adam([
            dict(params=[param for name, param in model.named_parameters() if name[-4:] == 'bias'],
                 lr=2 * Init_lr),
            dict(params=[param for name, param in model.named_parameters() if name[-4:] != 'bias'],
                 lr=Init_lr, weight_decay=weight_decay),
        ], betas=(momentum, 0.999)),
        'sgd': optim.SGD([
            dict(params=[param for name, param in model.named_parameters() if name[-4:] == 'bias'],
                 lr=2 * Init_lr),
            dict(params=[param for name, param in model.named_parameters() if name[-4:] != 'bias'],
                 lr=Init_lr, weight_decay=weight_decay),
        ], momentum=momentum, nesterov=True),
    }[optimizer_type]
    return optimizer


# def get_opt_and_scheduler(optimizer, Total_epoch=100, lr_decay_type: str = 'default'):
#     # ------------------------------------------------------------------#
#     #   lr_decay_type   使用到的学习率下降方式，可选的有'step'、'cos'
#     # ------------------------------------------------------------------#
#     scheduler = {
#         'cos': CosineLRScheduler(optimizer, t_initial=int(Total_epoch / 3), t_mul=1.0, lr_min=,
#                                  decay_rate=0.9, warmup_t=0, warmup_lr_init=, cycle_limit=5),
#         # when gamma=0.9, lr = 0.5 if epoch < 30; lr= 0.45 if 30 <= epoch < 60; lr = 0.405 if 60 <= epoch < 90
#         'steplr': StepLRScheduler(optimizer, decay_rate=0.9, decay_t=int(Total_epoch / 3), warmup_t=10,
#                                   warmup_lr_init=),  # use before train and after epoch line ,schuduler.step()
#         'plateau': PlateauLRScheduler(optimizer, decay_rate=0.9, patience_t=5, lr_min=Min_lr, threshold=1e-4),
#         # use after validation, scheduler.step(val_loss)
#         'expone': ExponentialLR(optimizer, gamma=0.9, last_epoch=-1),
#         'poly': Poly(optimizer, Init_lr, 0.9, Total_epoch),
#         'default': None
#     }[lr_decay_type]
#     return scheduler


'''Scaler
# ---------------------------------------------------------------------#
#   fp16        是否使用混合精度训练,可减少约一半的显存、需要pytorch1.7.1以上
#   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
# ---------------------------------------------------------------------#
'''


def get_scaler(use_amp):
    if use_amp:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None
    return scaler


# use Model Exponential Moving Average or not
def get_EMA(useEMA, net):
    if useEMA:
        model_ema_decay = 0.999
        model_ema_force_cpu = False
        from timm.utils import ModelEma, get_state_dict
        ema = ModelEma(net, decay=model_ema_decay, device='cpu' if model_ema_force_cpu else '', resume='')
        print("Using EMA with decay = %.8f" % model_ema_decay)
    else:
        ema = None
    return ema


'''Loss
# -----------------------------------------------------------------#
# focal loss用于防止正负样本不平衡。
# dice loss建议种类少或种类多但batchsize大时，设定为True；种类多但batchsize较小，设定为False
# 其他可选的Loss:
#             MSELoss(reduction="mean")
#             TverskyLoss(mode="multiclass")
#             JaccardLoss(mode="multiclass")
# -------------------------------------------------------------------#
'''


def get_loss(loss_name: str):
    losses = loss_name.split('+')
    return losses


# def use_ce(mixup=None, smoothing=0.):
#     if mixup is not None:
#         return SoftTargetCrossEntropy()
#     elif smoothing > 0.:
#         return LabelSmoothingCrossEntropy(smoothing=smoothing)
#     else:
#         return CrossEntropyLoss()


def get_criterion(loss_name, device_id, is_gpu=True, mode="binary"):
    losses = {
        "bce": BCELoss(),
        'bcew': BCEWithLogitsLoss(),
        'focal': FocalLoss(mode=mode),
        'dice': DiceLoss(mode=mode),
        'jaccard': JaccardLoss(mode=mode),
        'lovasz': LovaszLoss(mode=mode),
        # 'ce': use_ce(mixup, smoothing)
    }
    loss_names = get_loss(loss_name)
    loss_name = loss_names[0]
    criterion = losses[loss_name]
    if len(loss_names) > 1:
        for name in loss_names[1:]:
            loss = losses[name]
            criterion = JointLoss(criterion, loss, first_weight=1., second_weight=1.)
    if is_gpu:
        criterion = criterion.cuda(device=device_id)
    # if 'ce' not in loss_names:
    #     mixup = None
    # else:
    #     print("Mixup is Activated!")
    return criterion


def initial_logger(file):
    import logging
    logger = logging.getLogger('log')
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(file)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, num=1):
        self.val = val
        self.avg = val
        self.count = num
        self.sum = val * num
        self.initialized = True

    def update(self, val, num=1):
        if not self.initialized:
            self.initialize(val, num)
        else:
            self.add(val, num)

    def add(self, val, num):
        self.val = val
        self.count += num
        self.sum += val * num
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def draw(Total_epoch, train_loss_total_epochs, valid_loss_total_epochs, epoch_indexSet, logs_path):
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt
    from utils.get_metric import smooth
    x = [i for i in range(Total_epoch)]
    fig = plt.figure(figsize=(24, 8))
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(x, smooth(train_loss_total_epochs, 0.6), label='train loss')
    ax.plot(x, smooth(valid_loss_total_epochs, 0.6), label='val loss')
    ax.set_xlabel('Epoch', fontsize=15)
    ax.set_ylabel('Loss', fontsize=15)
    ax.set_title('train curve', fontsize=15)
    ax.grid(True)
    plt.legend(loc='upper right', fontsize=15)

    epoch_lr = epoch_indexSet['learningRate']
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(x, epoch_lr, label='Learning Rate')
    ax.set_xlabel('Epoch', fontsize=15)
    ax.set_ylabel('Learning Rate', fontsize=15)
    ax.set_title('lr curve', fontsize=15)
    ax.grid(True)
    plt.legend(loc='upper right', fontsize=15)

    epoch_iou = epoch_indexSet['FwIoU']
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(x, epoch_iou, label="FwIoU")
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("FwIoU", fontsize=15)
    ax.set_title("FwIoU index", fontsize=15)
    ax.grid(True)
    plt.legend(loc='upper right', fontsize=15)

    epoch_mpa = epoch_indexSet['mPA']
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(x, epoch_mpa, label="mPA")
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("meanPA", fontsize=15)
    ax.set_title("meanPA index", fontsize=15)
    ax.grid(True)
    plt.legend(loc='upper right', fontsize=15)
    plt.tight_layout()
    plt.savefig(logs_path + "./train_val.png")
    print("save plot in ", logs_path)


