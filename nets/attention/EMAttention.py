# encoding:utf-8
"""
    @Author: DorisFawkes
    @File:
    @Date: 2022/08/01 20:18
    @Description: The Expectation-Maximization Attention, ICCV 2019 Oral
    @PaperTitle: EMANet：Expectation-Maximization Attention Networks for Semantic Segmentation
"""
import math
from functools import partial

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision.models import resnet50


def _l2norm(inp, dim):
    '''Normlize the inp tensor with l2-norm.
    Returns a tensor where each sub-tensor of input along the given dim is
    normalized such that the 2-norm of the sub-tensor is equal to 1.
    Arguments:
        inp (tensor): The input tensor.
        dim (int): The dimension to slice over to get the ssub-tensors.
    Returns:
        (tensor) The normalized tensor.
    '''
    return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))


class EMAU(nn.Module):
    '''The Expectation-Maximization Attention Unit (EMAU).
        Arguments:
            c (int): The input and output channel number.
            k (int): The number of the bases.
            stage_num (int): The iteration number for EM.
        '''

    def __init__(self, c, k, stage_num=3):
        super(EMAU, self).__init__()
        self.stage_num = stage_num

        mu = torch.Tensor(1, c, k)
        mu.normal_(0, math.sqrt(2. / k))  # Init with Kaiming Norm.
        mu = _l2norm(mu, dim=1)
        self.register_buffer('mu', mu)

        self.conv1 = nn.Conv2d(c, c, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c, momentum=3e-4))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        idn = x
        # The first 1x1 conv
        x = self.conv1(x)

        # The EM Attention
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)  # b * c * n
        mu = self.mu.repeat(b, 1, 1)  # b * c * k
        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x.permute(0, 2, 1)  # b * n * c
                z = torch.bmm(x_t, mu)  # b * n * k
                z = F.softmax(z, dim=2)  # b * n * k
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x, z_)  # b * c * k
                mu = _l2norm(mu, dim=1)

        # !!! The moving averaging operation is writtern in training.py, which is significant.

        z_t = z.permute(0, 2, 1)  # b * k * n
        x = mu.matmul(z_t)  # b * c * n
        x = x.view(b, c, h, w)  # b * c * h * w
        x = F.relu(x, inplace=True)

        # The second 1x1 conv
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x, inplace=True)

        return x, mu


class EMANet(nn.Module):
    """ Implementation of EMANet (ICCV 2019 Oral)."""

    def __init__(self, n_classes):
        super().__init__()
        backbone = resnet50(pretrained=False)
        self.extractor = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4)

        # self.fc0 = ConvBNReLU(2048, 512, 3, 1, 1, 1)
        self.fc0 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.emau = EMAU(512, 64, 3)
        self.fc1 = nn.Sequential(
            # ConvBNReLU(512, 256, 3, 1, 1, 1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.Dropout2d(p=0.1))
        self.fc2 = nn.Conv2d(256, n_classes, 1)

        # Put the criterion inside the model to make GPU load balanced
        self.crit = CrossEntropyLoss(ignore_index=255,
                                     reduction='none')

    def forward(self, img, lbl=None, size=None):
        x = self.extractor(img)
        x = self.fc0(x)
        x, mu = self.emau(x)
        x = self.fc1(x)
        x = self.fc2(x)

        if size is None:
            size = img.size()[-2:]
        pred = F.interpolate(x, size=size, mode='bilinear', align_corners=True)

        if self.training and lbl is not None:
            loss = self.crit(pred, lbl)
            return loss, mu
        else:
            return pred


if __name__ == '__main__':
    model = EMANet(3)
    data = torch.randn(1, 3, 512, 512)
    print(model(data).size())