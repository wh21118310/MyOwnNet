# -*- coding: utf-8 -*-

"""
@Time : 2022/5/22
@Author : FaweksLee
@File : get_metric
@Description : 
"""
# from .eval_segm import frequency_weighted_IU, mean_accuracy
from .metrcs import SegmentationMetric


def smooth(v, w=0.85):
    last = v[0]
    smoothed = []
    for point in v:
        smoothed_val = last * w + (1 - w) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def binary_accuracy(pred, label):
    valid = (label < 2)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def FWIoU(pred, label, bn_mode=False, ignore_zero=False, num_classes=3):
    if bn_mode:
        pred = (pred >= 0.5)
        label = (label >= 0.5)
    elif ignore_zero:
        pred = pred - 1
        label = label - 1
    metric = SegmentationMetric(num_classes)
    metric.addBatch(pred, label)
    # FWIoU = frequency_weighted_IU(pred, label)
    # return FWIoU
    return metric.Frequency_Weighted_Intersection_over_Union()


def Acc(pred, label, bn_mode=False, ignore_zero=False, num_classes=3):
    if bn_mode:
        pred = (pred >= 0.5)
        label = (label >= 0.5)
    elif ignore_zero:
        pred = pred - 1
        label = label - 1
    # acc = mean_accuracy(pred, label)
    # return acc
    metric = SegmentationMetric(num_classes)
    metric.addBatch(pred, label)
    return metric.meanPixelAccuracy()
