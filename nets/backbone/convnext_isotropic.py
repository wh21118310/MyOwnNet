# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from torchinfo import summary


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super(Block, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        # Regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = rearrange(x, "n c h w -> n h w c")
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        # x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = rearrange(x, "n h w c -> n c h w")
        x = input + self.drop_path(x)
        return x


class ConvNeXtIsotropic(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
        https://arxiv.org/pdf/2201.03545.pdf
        Isotropic ConvNeXts (Section 3.3 in paper)

    Args:
        in_chans (int): Number of input images channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depth (tuple(int)): Number of blocks. Default: 18.
        dims (int): Feature dimension. Default: 384
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, depth=18, dim=384, drop_path_rate=0., layer_scale_init_value=0):
        super(ConvNeXtIsotropic, self).__init__()

        self.stem = nn.Conv2d(in_chans, dim, kernel_size=16, stride=16)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[Block(dim=dim, drop_path=dp_rates[i],
                                            layer_scale_init_value=layer_scale_init_value)
                                      for i in range(depth)])

        self.norm = LayerNorm(dim, eps=1e-6)  # final norm layer
        self.apply(self._init_weights)

        # self.head = nn.Linear(dim, num_classes)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        # if isinstance(m, (nn.Conv2d, nn.Linear)):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return x


# class UpDecoder(nn.Module):
#     def __init__(self, out_chans=3, dim=384):
#         super(UpDecoder, self).__init__()
#         self.Result = nn.Sequential(
#             nn.ConvTranspose2d(dim, out_chans, 16, 16),
#             LayerNorm(out_chans, eps=1e-6, data_format="channels_first"),
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, x):
#         return self.Result(x)


@register_model
def convnext_isotropic_small(in_channels=3, pretrained=False, **kwargs):
    model = ConvNeXtIsotropic(in_chans=in_channels, depth=18, dim=384, **kwargs)
    if pretrained:
        url = 'https://dl.fbaipublicfiles.com/convnext/convnext_iso_small_1k_224_ema.pth'
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["backbone"], strict=False)
    return model


@register_model
def convnext_isotropic_base(in_channels=3, pretrained=False, **kwargs):
    model = ConvNeXtIsotropic(in_chans=in_channels, depth=18, dim=768, **kwargs)
    if pretrained:
        url = 'https://dl.fbaipublicfiles.com/convnext/convnext_iso_base_1k_224_ema.pth'
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["backbone"], strict=False)
    return model


@register_model
def convnext_isotropic_large(in_channels=3, pretrained=False, **kwargs):
    model = ConvNeXtIsotropic(in_chans=in_channels, depth=36, dim=1024, **kwargs)
    if pretrained:
        url = 'https://dl.fbaipublicfiles.com/convnext/convnext_iso_large_1k_224_ema.pth'
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["backbone"], strict=False)
    return model


# @register_model
# def convnext_isotropic_small(pretrained=False, **kwargs):
#     backbone = ConvNeXtIsotropic(depth=18, dim=384, **kwargs)
#     if pretrained:
#         url = 'https://dl.fbaipublicfiles.com/convnext/convnext_iso_small_1k_224_ema.pth'
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
#         backbone.load_state_dict(checkpoint["backbone"], strict=False)
#     return backbone
#
#
# @register_model
# def convnext_isotropic_base(pretrained=False, **kwargs):
#     backbone = ConvNeXtIsotropic(depth=18, dim=768, **kwargs)
#     if pretrained:
#         url = 'https://dl.fbaipublicfiles.com/convnext/convnext_iso_base_1k_224_ema.pth'
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
#         backbone.load_state_dict(checkpoint["backbone"], strict=False)
#     return backbone
#
#
# @register_model
# def convnext_isotropic_large(pretrained=False, **kwargs):
#     backbone = ConvNeXtIsotropic(depth=36, dim=1024, **kwargs)
#     if pretrained:
#         url = 'https://dl.fbaipublicfiles.com/convnext/convnext_iso_large_1k_224_ema.pth'
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
#         backbone.load_state_dict(checkpoint["backbone"], strict=False)
#     return backbone
if __name__ == '__main__':
    data = torch.rand((4, 3, 512, 512))
    model = convnext_isotropic_base(3)
    # layers = list(model.children())
    # print(layers)
    print(summary(model, (4, 3, 512, 512)))
