# -*- coding: utf-8 -*-

"""
@Time : 2022/5/15
@Author : FaweksLee
@File : arguments
@Description : The arguments
"""
import argparse
import os
import random
from itertools import chain
from os.path import join

import cv2
import numpy as np
import torch
from pytorch_toolbelt.losses import JointLoss
from segmentation_models_pytorch.losses import *
from segmentation_models_pytorch.utils.losses import CrossEntropyLoss
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from torch import optim
from torch.nn import BCELoss, MSELoss, BCEWithLogitsLoss
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
from timm.scheduler import CosineLRScheduler


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


def distributedTraining(distributed: bool):
    from torch.cuda import device_count
    import torch.distributed as dist
    ngpus_per_node = device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print("[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
        sync_bn = get_sync(distributed)
        return local_rank, device, ngpus_per_node, sync_bn
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        return local_rank, device, ngpus_per_node


def model_load(model, model_path, local_rank, device):
    if model_path != "":
        if local_rank == 0:
            print('Load weights {}\n'.format(model_path))
        # ------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        # ------------------------------------------------------#
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, mpa_location=device)
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
    return model


def ModelToCuda(Cuda, distributed, model, local_rank):
    from torch.backends import cudnn
    from torch.nn import DataParallel
    from torch.nn.parallel import DistributedDataParallel
    if Cuda:
        if distributed:
            model = model.cuda(local_rank)
            model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model = DataParallel(model)
            cudnn.benchmark = True  # 当模型架构保持不变以及输入大小保持不变时可使用
            model = model.cuda()
    return model


'''
Training Settings.
'''
# --------------------------------------------------------------------#
# 训练分两个阶段，分别冻结与解冻阶段。设定冻结阶段是为了满足机器性能不足的同学的训练需求。
# 冻结训练需要的显存小，下那块非常差的情况下，可设定Freeze_Epoch等于UnFreeze_Epoch，此时仅进行冻结训练。
#   在此提供若干参数设置建议，各位训练者根据自己的需求进行灵活调整：
#   （一）从整个模型的预训练权重开始训练：
#       Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True
#           Adam:optimizer_type = 'adam'，Init_lr = 5e-4，weight_decay = 0。（冻结）
#           SGD:optimizer_type = 'sgd'，Init_lr = 7e-3，weight_decay = 1e-4。（冻结）
#       Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False
#           Adam:optimizer_type = 'adam'，Init_lr = 5e-4，weight_decay = 0。（不冻结）
#           SGD:optimizer_type = 'sgd'，Init_lr = 7e-3，weight_decay = 1e-4。（不冻结）
#       其中：UnFreeze_Epoch可以在100-300之间调整。
#   （二）从主干网络的预训练权重开始训练：
#       Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True
#           Adam:optimizer_type = 'adam'，Init_lr = 5e-4，weight_decay = 0。（冻结）
#           SGD：optimizer_type = 'sgd'，Init_lr = 7e-3，weight_decay = 1e-4。（冻结）
#       Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，
#           Adam:optimizer_type = 'adam'，Init_lr = 5e-4，weight_decay = 0。（不冻结）
#           SGD:optimizer_type = 'sgd'，Init_lr = 7e-3，weight_decay = 1e-4。（不冻结）
#       其中：由于从主干网络的预训练权重开始训练，主干的权值不一定适合语义分割，需要更多的训练跳出局部最优解。
#             UnFreeze_Epoch可以在120-300之间调整。
#             Adam相较于SGD收敛的快一些。因此UnFreeze_Epoch理论上可以小一点，但依然推荐更多的Epoch。
#   （三）batch_size的设置：
#       在显卡能够接受的范围内，以大为好。显存不足与数据集大小无关，提示显存不足（OOM或者CUDA out of memory）请调小batch_size。
#       正常情况下Freeze_batch_size建议为Unfreeze_batch_size的1-2倍。不建议设置的差距过大，因为关系到学习率的自动调整。
# ----------------------------------------------------------------------------------------------------------------------------#
# ------------------------------------------------------------------#
#   冻结阶段训练参数
#   此时模型的主干被冻结了，特征提取网络不发生改变。占用的显存较小，仅对网络进行微调
# ------------------------------------------------------------------#
Freeze_Train = False
Init_Epoch = 0  # 模型初始训练轮次，其值可大于Freeze_Epoch，如:Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
Freeze_Epoch = 50  # 模型冻结训练的Freeze_Epoch (当Freeze_Train=False时失效)
Freeze_batch_size = 8  # 模型冻结训练的batch_size (当Freeze_Train=False时失效)
# ------------------------------------------------------------------#
#   解冻阶段训练参数
#   此时模型的主干不被冻结了，特征提取网络会发生改变。占用的显存较大，网络所有的参数都会发生改变
# ------------------------------------------------------------------#
Total_Epoch = 100  # 模型总共的训练epoch
Unfreeze_batch_size = 10  # 模型解冻后的batch_size
# -------------------------------------------------------------------#
#   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
# -------------------------------------------------------------------#
batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

'''Optimizer & Scheduler
# ------------------------------------------------------------------#
#   optimizer_type  使用到的优化器种类，可选的有adam、sgd
#   momentum        优化器内部使用到的momentum参数
#   weight_decay    权值衰减，可防止过拟合。adam会导致weight_decay错误，使用adam时建议设置为0。
# ------------------------------------------------------------------#
'''


def get_opt_and_scheduler(model, optimizer_type: str, lr_decay_type: str, momentum: float, Total_epoch=100):
    weight_decay = {"sgd": 1e-4, "adam": 0}[optimizer_type]
    # ------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率，建议设定:
    #                       Adam:  Init_lr=5e-4
    #                       SGD:   Init_lr=7e-3
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    # ------------------------------------------------------------------#
    Init_lr = {"sgd": 7e-3, "adam": 5e-4}[optimizer_type]
    Min_lr = Init_lr * 0.1
    optimizer = {
        'adam': optim.Adam(chain(model.parameters()), Init_lr, betas=(momentum, 0.999),
                           weight_decay=weight_decay),
        'sgd': optim.SGD(chain(model.parameters()), Init_lr, momentum=momentum, nesterov=True,
                         weight_decay=weight_decay),
    }[optimizer_type]
    # ------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有'step'、'cos'
    # ------------------------------------------------------------------#
    scheduler = {
        'cos': CosineLRScheduler(optimizer, t_initial=100, t_mul=1.0, lr_min=Min_lr,
                                 decay_rate=0.95, warmup_t=0, warmup_lr_init=Init_lr * 0.5, cycle_limit=10),
        "cosW": CosineAnnealingWarmRestarts(optimizer, T_0=int(Total_epoch / 3), T_mult=1, eta_min=Min_lr,
                                            last_epoch=-1),
        # lr = 0.05 if epoch < 30; lr= 0.005 if 30 <= epoch < 60; lr = 0.0005 if 60 <= epoch < 90
        'steplr': StepLR(optimizer, step_size=50, gamma=0.9)
    }[lr_decay_type]
    return optimizer, scheduler


'''Scaler
# ---------------------------------------------------------------------#
#   fp16        是否使用混合精度训练,可减少约一半的显存、需要pytorch1.7.1以上
#   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
# ---------------------------------------------------------------------#
'''


def get_scaler(fp16):
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None
    return scaler


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


def use_ce(mixup=None, smoothing=0.):
    if mixup is not None:
        return SoftTargetCrossEntropy()
    elif smoothing > 0.:
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    else:
        return CrossEntropyLoss()


def get_criterion(loss_name, mixup=None, smoothing=0., is_gpu=True, mode="binary"):
    losses = {
        "bce": BCELoss(),
        'bcew': BCEWithLogitsLoss(),
        'focal': FocalLoss(mode=mode),
        'dice': DiceLoss(mode=mode),
        'jaccard': JaccardLoss(mode=mode),
        'lovasz': LovaszLoss(mode=mode),
        'ce': use_ce(mixup, smoothing)
    }
    loss_names = get_loss(loss_name)
    loss_name = loss_names[0]
    criterion = losses[loss_name]
    if len(loss_names) > 1:
        for name in loss_names[1:]:
            loss = losses[name]
            criterion = JointLoss(criterion, loss, first_weight=1., second_weight=1.)
    if is_gpu:
        criterion = criterion.cuda()
    if 'ce' not in loss_names:
        mixup = None
    else:
        print("Mixup is Activated!")
    return criterion, mixup


'''MixUp
#----------------------------
# Used from Timm, Setting from ConvNeXt
#---------------------------
'''


def get_mixup(mixup=0.8, cutmix=1.0, cutmix_minmax=None, mixup_prob=1.0, mixup_switch_prob=0.5, mixup_mode='batch',
              smoothing=0.1, num_classes=2):
    from timm.data import Mixup
    mixup_active = mixup > 0 or cutmix > 0. or cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(mixup_alpha=mixup, cutmix_alpha=cutmix, cutmix_minmax=cutmix_minmax, prob=mixup_prob,
                         switch_prob=mixup_switch_prob, mode=mixup_mode, label_smoothing=smoothing,
                         num_classes=num_classes)
    else:
        mixup_fn = None
    return mixup_fn


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

    def initialize(self, val, count, weight):
        self.val = val
        self.avg = val
        self.count = count
        self.sum = val * weight
        self.initialized = True

    def update(self, val, count=1, weight=1):
        if not self.initialized:
            self.initialize(val, count, weight)
        else:
            self.add(val, count, weight)

    def add(self, val, count, weight):
        self.val = val
        self.count += count
        self.sum += val * weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg



def draw(Total_epoch, train_loss_total_epochs, valid_loss_total_epochs, epoch_lr, epoch_iou, logs_path):
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt
    from utils.get_metric import smooth
    x = [i for i in range(Total_epoch)]
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(x, smooth(train_loss_total_epochs, 0.6), label='train loss')
    ax.plot(x, smooth(valid_loss_total_epochs, 0.6), label='val loss')
    ax.set_xlabel('Epoch', fontsize=15)
    ax.set_ylabel('Loss', fontsize=15)
    ax.set_title('train curve', fontsize=15)
    ax.grid(True)
    plt.legend(loc='upper right', fontsize=15)

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(x, epoch_lr, label='Learning Rate')
    ax.set_xlabel('Epoch', fontsize=15)
    ax.set_ylabel('Learning Rate', fontsize=15)
    ax.set_title('lr curve', fontsize=15)
    ax.grid(True)
    plt.legend(loc='upper right', fontsize=15)

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(x, epoch_iou, label="FwIoU")
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("FwIoU", fontsize=15)
    ax.set_title("FwIoU index", fontsize=15)
    ax.grid(True)
    plt.legend(loc='upper right', fontsize=15)
    plt.tight_layout()
    plt.savefig(logs_path + "./train_val.png")
    print("save plot in ", logs_path)


import torch.nn.functional as F


def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()