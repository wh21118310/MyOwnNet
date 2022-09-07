# -*- coding: utf-8 -*-

"""
@Time : 2022/5/26
@Author : FaweksLee
@File : metrcs
@Description : 
"""
import torch

"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
import numpy as np

__all__ = ['SegmentationMetric']

"""
confusionMetric
PL     P     N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)
        self.eps = 1e-8

    def get_tp_fp_tn_fn(self):
        tp = np.diag(self.confusionMatrix)
        fp = self.confusionMatrix.sum(axis=0) - np.diag(self.confusionMatrix)
        fn = self.confusionMatrix.sum(axis=1) - np.diag(self.confusionMatrix)
        tn = np.diag(self.confusionMatrix).sum() - np.diag(self.confusionMatrix)
        return tp, fp, tn, fn

    def Precision(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        precision = tp / (tp + fp)
        return precision

    def Recall(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        recall = tp / (tp + fn)
        return recall

    def F1(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F1 = (2.0 * precision * recall) / (precision + recall)
        return F1

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / (self.confusionMatrix.sum() + self.eps)
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=0) + self.eps)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def Dice(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Dice = 2 * tp / ((tp + fp) + (tp + fn))
        return Dice

    def IoU(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        IoU = tp / (tp + fn + fp + self.eps)
        return IoU

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        # intersection = np.diag(self.confusionMatrix)
        # union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
        #     self.confusionMatrix)
        # IoU = intersection / union
        iou = self.IoU()
        mIoU = np.nanmean(iou)
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt images and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIoU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / (np.sum(self.confusionMatrix) + self.eps)
        # iu = np.diag(self.confusionMatrix) / (np.sum(self.confusionMatrix, axis=1) +
        #                                       np.sum(self.confusionMatrix, axis=0) -
        #                                       np.diag(self.confusionMatrix))
        iou = self.IoU()
        FWIoU = (freq[freq > 0] * iou[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape, 'Predict shape:{}, label shape:{}'.format(imgPredict.shape,
                                                                                             imgLabel.shape)
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


if __name__ == '__main__':
    imgPredict = torch.rand(4, 3, 12, 12)
    imgLabel = torch.rand(4, 3, 12, 12)
    metric = SegmentationMetric(3)
    metric.addBatch(imgPredict, imgLabel)
    acc = metric.pixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    print(acc, mIoU)
