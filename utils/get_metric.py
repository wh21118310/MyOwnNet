# -*- coding: utf-8 -*-

"""
@Time : 2022/5/22
@Author : FaweksLee
@File : get_metric
@Description : 
"""
from .metrcs import SegmentationMetric


def smooth(v, w=0.85):
    last = v[0]
    smoothed = []
    for point in v:
        smoothed_val = last * w + (1 - w) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


# def OverallAccuracy(pred, label):
#     valid = (label < 2)
#     acc_sum = (valid * (pred == label)).sum()
#     valid_sum = valid.sum()
#     acc = float(acc_sum) / (valid_sum + 1e-10)
#     return acc, valid_sum


def FWIoU(pred, label, bn_mode=False, ignore_zero=False, num_classes=2):
    if bn_mode:
        pred = (pred >= 0.5)
        label = (label >= 0.5)
    elif ignore_zero:
        pred = pred - 1
        label = label - 1
    metric = SegmentationMetric(num_classes)
    metric.addBatch(pred, label)
    return metric.Frequency_Weighted_Intersection_over_Union()


def mPA(pred, label, bn_mode=False, ignore_zero=False, num_classes=2):
    if bn_mode:
        pred = (pred >= 0.5)
        label = (label >= 0.5)
    elif ignore_zero:
        pred = pred - 1
        label = label - 1
    metric = SegmentationMetric(num_classes)
    metric.addBatch(pred, label)
    return metric.meanPixelAccuracy()


def PA(pred, label, bn_mode=False, ignore_zero=False, num_classes=2):
    if bn_mode:
        pred = (pred >= 0.5)
        label = (label >= 0.5)
    elif ignore_zero:
        pred = pred - 1
        label = label - 1
    metric = SegmentationMetric(num_classes)
    metric.addBatch(pred, label)
    return metric.pixelAccuracy()


def Dice(pred, label, bn_mode=False, ignore_zero=False, num_classes=2):
    if bn_mode:
        pred = (pred >= 0.5)
        label = (label >= 0.5)
    elif ignore_zero:
        pred = pred - 1
        label = label - 1
    metric = SegmentationMetric(num_classes)
    metric.addBatch(pred, label)
    return metric.Dice()


def Precision(pred, label, bn_mode=False, ignore_zero=False, num_classes=2):
    if bn_mode:
        pred = (pred >= 0.5)
        label = (label >= 0.5)
    elif ignore_zero:
        pred = pred - 1
        label = label - 1
    metric = SegmentationMetric(num_classes)
    metric.addBatch(pred, label)
    return metric.Precision()


def Recall(pred, label, bn_mode=False, ignore_zero=False, num_classes=2):
    if bn_mode:
        pred = (pred >= 0.5)
        label = (label >= 0.5)
    elif ignore_zero:
        pred = pred - 1
        label = label - 1
    metric = SegmentationMetric(num_classes)
    metric.addBatch(pred, label)
    return metric.Recall()


def F1(pred, label, bn_mode=False, ignore_zero=False, num_classes=2):
    if bn_mode:
        pred = (pred >= 0.5)
        label = (label >= 0.5)
    elif ignore_zero:
        pred = pred - 1
        label = label - 1
    metric = SegmentationMetric(num_classes)
    metric.addBatch(pred, label)
    return metric.F1()
