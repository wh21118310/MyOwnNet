# -*- coding: utf-8 -*-

"""
    @Time : 2022/10/6 15:12
    @Author : FaweksLee
    @Email : 121106010719@njust.edu.cn
    @File : CVF_denoising
    @Description : 模型复现
"""
import torch
from torch import nn


class GenClean(nn.Module):
    def __init__(self, channels=3, numLayers=17, features=64):
        super(GenClean, self).__init__()
        kernel, padding = 3, 1
        layers = list()
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel, padding=padding))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(numLayers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel, padding=padding))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1, padding=0))
        self.genclean = nn.Sequential(*layers)
        for m in self.genclean:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, x):
        out = self.genclean(x)
        return out


class GenNoise(nn.Module):
    def __init__(self, NLayer=10, Fsize=64, channels=3):
        super(GenNoise, self).__init__()
        kernel, padding = 3, 1
        m = [nn.Conv2d(channels, Fsize, kernel_size=kernel, padding=padding),
             nn.ReLU(inplace=True)]
        for i in range(NLayer - 1):
            m.append(nn.Conv2d(Fsize, Fsize, kernel_size=kernel, padding=padding))
            m.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*m)

        gen_noise_w = []
        for i in range(4):
            gen_noise_w.append(nn.Conv2d(Fsize, Fsize, kernel_size=kernel, padding=padding))
            gen_noise_w.append(nn.ReLU(inplace=True))
        gen_noise_w.append(nn.Conv2d(Fsize, 3, kernel_size=1, padding=0))
        self.gen_noise_w = nn.Sequential(*gen_noise_w)

        gen_noise_b = []
        for i in range(4):
            gen_noise_b.append(nn.Conv2d(Fsize, Fsize, kernel_size=kernel, padding=padding))
            gen_noise_b.append(nn.ReLU(inplace=True))
        gen_noise_b.append(nn.Conv2d(Fsize, 3, kernel_size=1, padding=0))
        self.gen_noise_b = nn.Sequential(*gen_noise_b)

        for m in self.body:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)
        for m in self.gen_noise_w:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)
        for m in self.gen_noise_b:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, x, weights=None, test=False):
        noise = self.body(x)
        noise_w = self.gen_noise_w(noise)
        noise_b = self.gen_noise_b(noise)
        m_w = torch.mean(torch.mean(noise_w, -1), -1).unsqueeze(-1).unsqueeze(-1)
        noise_w = noise_w - m_w
        m_b = torch.mean(torch.mean(noise_b, -1), -1).unsqueeze(-1).unsqueeze(-1)
        noise_b = noise_b - m_b
        return noise_w, noise_b


class CVF_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_colors = 3
        FSize = 64
        self.gen_noise = GenNoise(Fsize=FSize)
        self.genclean = GenClean()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, weights=None, test=False):
        clean = self.genclean(x)
        noise_w, noise_b = self.gen_noise(x - clean)
        return noise_w, noise_b, clean


if __name__ == '__main__':
    data = torch.randn((4, 3, 512, 512))
    model = CVF_model()
    noise_w, noise_b, clean = model(data)
    print(clean)
