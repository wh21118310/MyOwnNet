# -*- coding: utf-8 -*-

"""
    @Time : 2022/6/12
    @Author : FaweksLee
    @File : bk
    @Description : 
"""
import torch
from einops import rearrange
from torchinfo import summary
from torchvision.models.quantization import resnet50

from nets.backbone.Res2Net import res2net50_v1b_26w_4s
from nets.backbone.Swin_transformer import SwinNet
from nets.backbone.convnext import ConvNeXt_Seg
import torch.nn as nn

models_type1 = [
    'resnet50', 'res2net50', 'convnext_base',
]
models_type2 = [
    "swinT_base"
]


class Backbone(nn.Module):
    def __init__(self, bk="res2net50", model_path=None, in_channels=3):
        super(Backbone, self).__init__()
        self.backbone = bk
        if self.backbone == 'resnet50' or self.backbone == 'res2net50':
            if self.backbone == 'resnet50':
                self.backbone = resnet50(pretrained=False)
            elif self.backbone == 'res2net50':
                self.backbone = res2net50_v1b_26w_4s(pretrained=False)
            self.layer1 = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu,
                                        self.backbone.maxpool, self.backbone.layer1)
            self.layer2 = self.backbone.layer2
            self.layer3 = self.backbone.layer3
            self.layer4 = self.backbone.layer4
        elif self.backbone == 'convnext_base':
            self.backbone = ConvNeXt_Seg(in_channels, model_type="base", pretrained=False)
            self.downsample = self.backbone.downsample_layers
            self.stage = self.backbone.stages
            self.layer1 = nn.Sequential(self.downsample[0], self.stage[0], self.backbone.norm0)
            self.layer2 = nn.Sequential(self.downsample[1], self.stage[1], self.backbone.norm1)
            self.layer3 = nn.Sequential(self.downsample[2], self.stage[2], self.backbone.norm2)
            self.layer4 = nn.Sequential(self.downsample[3], self.stage[3], self.backbone.norm3)
        elif self.backbone == 'swinT_base':
            self.backbone = SwinNet(3, "base")
            # self.pos_drop = self.backbone.pos_drop
            # self.layers = self.backbone.layers
            # # self.final_norm = self.backbone.norm
            # # self.layer1 = self.layers[0]
            # self.layer1 = self.backbone.patch_embed
            # self.layer2 = self.layers[0]
            # self.layer3 = self.layers[1]
            # self.layer4_1 = self.layers[2]
            # self.layer4_2 = self.layers[3]

        # 是否载入预训练模型
        if model_path is not None:
            self.backbone.load_state_dict(torch.load(model_path), strict=False)

    def forward(self, x, model_type="resnet50"):
        if model_type in models_type1:
            layer1 = self.layer1(x)  # [-1, 256, h/4, w/4]
            layer2 = self.layer2(layer1)  # [-1, 512, h/8, w/8]
            layer3 = self.layer3(layer2)  # [-1, 1024, h/16, w/16]
            layer4 = self.layer4(layer3)  # [-1, 2048, h/32, w/32]
            return layer1, layer2, layer3, layer4
        elif model_type in models_type2:
            # x = self.layer1(x)
            # H, W = x.size()[-2], x.size()[-1]
            # x = self.pos_drop(x)
            # layer1 = x
            # # layer1 = rearrange(x, " b (h w) c-> b c h w", h=H, w=W)
            # x, H, W = self.layer2(x, H, W)
            # layer2 = x
            # # H, W = x.size()[-2], x.size()[-1]
            # # layer2 = rearrange(x, " b (h w) c-> b c h w", h=H, w=W)
            # x = self.layer3(x, H, W)
            # layer3 = x
            # # layer3 = rearrange(x, " b (h w) c-> b c h w", h=H, w=W)
            # x = self.layer4_1(x, H, W)
            # x = self.layer4_2(x, H, W)
            # # layer4 = rearrange(x, " b (h w) c-> b c h w", h=H, w=W)
            # layer4 = x
            layer1, layer2, layer3, layer4 = self.backbone(x)
            return layer1, layer2, layer3, layer4


if __name__ == '__main__':
    data = torch.rand((3, 3, 512, 512))
    net = Backbone(bk='swinT_base')
    result = net(data, model_type='swinT_base')
    # print(result)
