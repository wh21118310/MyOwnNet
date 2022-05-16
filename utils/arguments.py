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

import numpy as np
import torch.cuda

from MyOwnNet.utils.data_process import DataSetWithSupervised, DataSetWithNosupervised


def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_device():
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
    import torch.distributed as dist
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
    return Cuda, device, local_rank, sync_bn


def get_dataSet(DataPath: str, supervised=True):
    # Note1: remember to change the path of images and labels
    images = os.path.join(DataPath, "images")
    if supervised:
        labels = os.path.join(DataPath, "gt")
        # Note2: the tfs is refered to transform
        data = DataSetWithSupervised(imgs_dir=images, labels_dir=labels, tfs=None)
    else:
        data = DataSetWithNosupervised(imgs_dir=images, tfs=None)
    return data


def get_args_parser():
    # trainingPath = "../dataset"
    cuda, device, local_rank, sync_bn = get_device()
    params = {"cuda": cuda, "device": device, "local_rank": local_rank}
    if sync_bn:
        params.update({"sync_bn": sync_bn})

    return params


if __name__ == '__main__':
    get_args_parser()