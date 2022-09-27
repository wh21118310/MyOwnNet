import cv2
import numpy as np
import math

def lee_filter(image, kernal_size):
    data = np.float32(image) / 255.0
    # data = np.float32(image)
    # 计算原始数据的方差
    var_ref = np.var(data)
    kernal_size = 3
    indexer = kernal_size // 2
    # 边缘填充
    padding = np.zeros((data.shape[0] + indexer * 2, data.shape[1] + indexer * 2), data.dtype)
    padding[indexer:padding.shape[0] - indexer, indexer:padding.shape[1] - indexer] = data
    del data
    for i in range(indexer, padding.shape[0] - indexer):
        for j in range(indexer, padding.shape[1] - indexer):
            # 计算均值

            mean_k = np.mean(padding[i - indexer:i + indexer + 1, j - indexer:j + indexer + 1])
            kernal_var = np.var(padding[i - indexer:i + indexer + 1, j - indexer:j + indexer + 1])
            Weight = (kernal_var) / ((kernal_var) + (var_ref))
            c = padding[i, j]
            padding[i, j] = mean_k + (c - mean_k) * Weight
    # padding = np.clip(padding * 255, 0, 255).astype(np.uin )
    padding = np.clip(padding * 255.0, -99999, 99999).astype(np.float32)
    # a,b=padding.shape
    a=padding
    # print("shuchu",a,b)
    return padding
