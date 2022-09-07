# -*- coding: utf-8 -*-

"""
    @Time : 2022/9/7 20:50
    @Author : FaweksLee
    @Email : 121106010719@njust.edu.cn
    @File : ESPCN
    @Description :  Efficient Sub-Pixel Convolutional Neural Network
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class ESPCNet(nn.Module):
    def __init__(self, upscale_factor):
        super(ESPCNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(32, (upscale_factor ** 2), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # [b, 3, H, W]->[b, 64, H, W]
        x = torch.relu(self.conv2(x))  # [b, 64, H, W] -> [b, 32, H, W]
        x = self.conv3(x)  # [b, 32, H, W] -> [b, 9, H, W]
        x = self.pixel_shuffle(x)  # [b, 1, 3H, 3W]
        x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    model = ESPCNet(upscale_factor=3)
    print(model)
    oritensor = torch.randn(1, 3, 512, 512)
    oritensor = torch.clamp(oritensor, 0, 1)
    newtensor = torch.clamp(model(oritensor), 0, 1)
    orinumpy = oritensor.detach().squeeze().permute(1, 2, 0).numpy()
    newnumpy = newtensor.detach().squeeze(0).permute(1, 2, 0).numpy()
    plt.imshow(orinumpy)
    plt.title('original noise')
    plt.show()
    plt.imshow(newnumpy)
    plt.title('reconstruct noise')
    plt.show()
    print(orinumpy.shape)
    print(newnumpy.shape)
