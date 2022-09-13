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
        if pred.size() != mask.size():
            mask = torch.repeat_interleave(mask, pred.size()[1], dim=1)
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


'''Soft Dice'''


def soft_dice_score(output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None
                    ) -> torch.Tensor:
    """IoU, same as F-score"""
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score


'''Soft Jaccard'''


def soft_jaccard_score(
        output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None
) -> torch.Tensor:
    """Also same as IoU"""
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)  # intersection
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)

    union = cardinality - intersection  # union
    jaccard_score = (intersection + smooth) / (union + smooth).clamp_min(eps)  # intersection / union = IOU
    return jaccard_score


'''Soft Tversky'''


def soft_tversky_score(output: torch.Tensor, target: torch.Tensor, alpha: float, beta: float,
                       smooth: float = 0.0, eps: float = 1e-7, dims=None) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)  # TP
        fp = torch.sum(output * (1. - target), dim=dims)
        fn = torch.sum((1 - output) * target, dim=dims)
    else:
        intersection = torch.sum(output * target)  # TP
        fp = torch.sum(output * (1. - target))
        fn = torch.sum((1 - output) * target)
    tversky_score = (intersection + smooth) / (intersection + alpha * fp + beta * fn + smooth).clamp_min(eps)
    return tversky_score
