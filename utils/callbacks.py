# -*- coding: utf-8 -*-

"""
@Time : 2022/5/18
@Author : FaweksLee
@File : callbacks
@Description : 
"""
import logging


def initial_logger(file):
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
