# 前言

# 部分内容的实现

SSIM、PSNR可采用`skimage.measure`包实现。具体来说:
`skimage.metrics.peak_signal_noise_ratio
(im_true, im_test, data_range=None, dynamic_range=None)`

计算图像的峰值信噪比（PSNR）。

参数：

* `im_true`：ndarray地面真相图像。
* `im_test`：ndarray测试图像。
* `data_range`：int 输入图像的数据范围（最小和最大可能值之间的距离）。默认情况下，这是从图像数据类型估计的。

返回：

* `psnr`：float PSNR指标。

`skimage.metrics.structural_similarity(X, Y, win_size=None, gradient=False, data_range=None, multichannel=False, gaussian_weights=False, full=False, dynamic_range=None, **kwargs)`

计算两幅图像之间的平均结构相似性指数。

参数：

* `X，Y`：ndarray 图片。任何维度。
* `win_size`：int或None用于比较的滑动窗口的边长。必须是奇数值。如果gaussian_weights为True，则忽略它，窗口大小将取决于西格玛。
* `gradient`：布尔，可选如果为True ，也会返回渐变。
* `data_range`：int，可选输入图像的数据范围（最小和最大可能值之间的距离）。默认情况下，这是从图像数据类型估计的。
* `multichannel`：bool，可选如果为True，则将数组的最后一个维度视为通道。相似性计算是针对每个通道独立完成的，然后进行平均。
* `gaussian_weights`：bool，可选如果为True ，则每个补丁均具有由宽度为sigma = 1.5的归一化高斯内核进行空间加权的均值和方差。
* `full：bool`：可选如果为True

返回：

* `mssim`：float图像上的平均结构相似度。
* `grad`：ndarray X和Y R333之间的结构相似性指数的梯度。这仅在梯度设置为True时才会返回。
* `S`：ndarray完整的SSIM图像。这仅在full设置为True时才会返回。

其他参数：
* `use_sample_covariance`：bool如果为真，用N-1归一化协方差而不是N，其中N是滑动窗口内的像素数。
* `K1`：浮点算法参数，K1（小常量，参见R332）
* `K2`：浮点算法参数，K2（小常量，参见R332）
* `sigma`：当gaussian_weights为真时为高斯浮点数。

注意:
* 为了配合Wang等的实施。将gaussian_weights设置为True，将sigma设置为1.5，并将use_sample_covariance设置为False。