# -*- coding: utf-8 -*-

"""
    @Time : 2022/8/30 11:50
    @Author : FaweksLee
    @Email : 121106010719@njust.edu.cn
    @File : loss
    @Description : 
"""
import torch
import torch.nn.functional as F
from torch.nn import Module


class IoU(Module):
    def __init__(self):
        super(IoU, self).__init__()

    def _iou(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3)) - inter
        iou = 1 - (inter / union)
        return iou.mean()

    def forward(self, pred, target):
        return self._iou(pred, target)


###################################################################
# #################### structure loss #############################
###################################################################
class structure_loss(torch.nn.Module):
    """loss function(ref: F3Net - AAAI - 2020)"""
    def __init__(self):
        super(structure_loss, self).__init__()

    def _structure_loss(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce=False)
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - inter / (union - inter)
        return (wbce + wiou).mean()

    def forward(self, pred, mask):
        return self._structure_loss(pred, mask)
