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
import torch.cuda
import torch.distributed as dist
from pytorch_toolbelt.losses import JointLoss
from segmentation_models_pytorch.losses import *
from torch import optim
from torch.nn import BCELoss, MSELoss, BCEWithLogitsLoss
from torch.optim.lr_scheduler import StepLR
from timm.scheduler import CosineLRScheduler


# set seeds
def seed_torch(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


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
        os.makedirs(Path)
    else:
        pass


'''
1. CUDA、Parallel Setting
'''
# ---------------------------------#
#   Cuda    whether use Cuda
#           if no GPU, set this as False,Please!
# ---------------------------------#
Cuda = True
# ---------------------------------------------------------------------#
#   distributed     whether to use a single MultiGPU distributed running system
#                   Terminal command just support Ubuntu. CUDA_VISIBLE_DEVICES used to specify a GPU under Ubuntu
#                   All GPUs are invoked in DP mode by default in Windows, DDP is not Supported.
#   DP Mode：
#       Set                  distributed = False
#       Input in Terminal    CUDA_VISIBLE_DEVICES=0,1 python train.py
#   DDP Mode：
#       Set                  distributed = True
#       Input in Terminal    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
# ---------------------------------------------------------------------#
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ngpus_per_node = torch.cuda.device_count()
distributed = False
if distributed:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", local_rank)
    if local_rank == 0:
        print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
        print("Gpu Device Count : ", ngpus_per_node)
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = 0
# ---------------------------------------------------------------------#
#   sync_bn     whether to use sync_bn in DDP multiGPU mode
# ---------------------------------------------------------------------#
if distributed:
    sync_bn = True
else:
    sync_bn = False

'''
2. Other Settings.
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
Total_Epoch = 1  # 模型总共的训练epoch
Unfreeze_batch_size = 20  # 模型解冻后的batch_size
# -------------------------------------------------------------------#
#   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
# -------------------------------------------------------------------#
batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size


# ------------------------------------------------------------------#
#   optimizer_type  使用到的优化器种类，可选的有adam、sgd
#   momentum        优化器内部使用到的momentum参数
#   weight_decay    权值衰减，可防止过拟合。adam会导致weight_decay错误，使用adam时建议设置为0。
# ------------------------------------------------------------------#
def get_opt_and_scheduler(model, optimizer_type: str, lr_decay_type: str, momentum: float, Total_epoch=100):
    weight_decay = {"sgd": 1e-4, "adam": 0}[optimizer_type]
    # ------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率，建议设定:
    #                       Adam:  Init_lr=5e-4
    #                       SGD:   Init_lr=7e-3
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    # ------------------------------------------------------------------#
    Init_lr = {"sgd": 7e-2, "adam": 5e-3}[optimizer_type]
    Min_lr = Init_lr * 0.01
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
        'cos': CosineLRScheduler(optimizer, t_initial=10, t_mul=1.0, lr_min=Min_lr,
                                 decay_rate=1, warmup_t=0, warmup_lr_init=Init_lr, cycle_limit=10),
        # lr = 0.05 if epoch < 30; lr= 0.005 if 30 <= epoch < 60; lr = 0.0005 if 60 <= epoch < 90
        'steplr': StepLR(optimizer, step_size=int(Total_epoch / 3), gamma=0.9)
    }[lr_decay_type]
    return optimizer, scheduler


# ---------------------------------------------------------------------#
#   fp16        是否使用混合精度训练,可减少约一半的显存、需要pytorch1.7.1以上
#   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
# ---------------------------------------------------------------------#
def get_scaler(fp16):
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None
    return scaler


# -----------------------------------------------------------------#
# focal loss用于防止正负样本不平衡。dice loss建议种类少或种类多但batchsize大时，设定为True；种类多但batchsize较小，设定为False
# 其他可选的Loss:
#             MSELoss(reduction="mean")
#             TverskyLoss(mode="multiclass")
#             JaccardLoss(mode="multiclass")
# -------------------------------------------------------------------#
def get_loss(loss_name: str):
    losses = loss_name.split('+')
    return losses


# def get_criterion(base: str, focal: bool, dice: bool, is_gpu=True, mode="binary"):
def get_criterion(loss_name, is_gpu=True, mode="binary"):
    losses = {
        "bce": BCELoss(),
        'bcew': BCEWithLogitsLoss(),
        'focal': FocalLoss(mode=mode),
        'dice': DiceLoss(mode=mode),
        'jaccard': JaccardLoss(mode=mode),
        'lovasz': LovaszLoss(mode=mode)
    }
    loss_names = get_loss(loss_name)
    if len(loss_names) == 1:
        loss_name = loss_names[0]
        criterion = losses[loss_name]
    else:
        loss = loss_names[0]
        criterion = losses[loss]
        for name in loss_names[1:]:
            loss = losses[name]
            criterion = JointLoss(criterion, loss, first_weight=1., second_weight=1.)
    if is_gpu:
        criterion = criterion.cuda()
    return criterion


def get_args_parser():
    # trainingPath = "../dataset"
    params = dict()
    GPU_Settings = {"cuda": Cuda, "device": device, "local_rank": local_rank, "GPU_Count": ngpus_per_node,
                    "distributed": distributed}
    if sync_bn:
        GPU_Settings.update({"sync_bn": sync_bn})
    params.update(GPU_Settings)
    # Setting the training about FreeTraining、Init_Epoch、Total_Epoch、UnFreeze_Batch etc.
    Training_Settings = {"Freeze": Freeze_Train, "Init_Epoch": Init_Epoch, "Total_Epoch": Total_Epoch,
                         "UnFreeze_Batch": Unfreeze_batch_size, "batch_size": batch_size}
    if Freeze_Train:
        Training_Settings.update({"Freeze_Epoch": Freeze_Epoch, "Freeze_batch": Freeze_batch_size})
    params.update(Training_Settings)
    return params


if __name__ == '__main__':
    params = get_args_parser()
    print(params)
    # losses = "bcew+focal+dice"
    # loss = get_criterion(losses)
    # print(loss)
