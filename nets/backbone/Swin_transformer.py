# -*- coding: utf-8 -*-

"""
@Time : 2022/5/6
@Author : MSRA
@File : Swin_transformer
@Description :
    A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    - https://arxiv.org/pdf/2103.14030
    Code/weights from https://github.com/microsoft/Swin-Transformer
    Copyright (c) 2021 Microsoft
    Licensed under The MIT License
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from typing import Optional
from einops import rearrange


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分为没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//window_size, window_size, W//window_size, window_size, C] -> [B, H//window_size, W//window_size,
    # window_size,
    # window_size, C]
    # view: [B, H//window_size, W//window_size, window_size, window_size, C] -> [B*num_windows, window_size,
    # window_size, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of images
        W (int): Width of images
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, window_size, window_size, C] -> [B, H//window_size, W//window_size, window_size,
    # window_size, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//window_size, W//window_size, window_size, window_size, C] -> [B, H//window_size, window_size,
    # W//window_size,
    # window_size, C]
    # view: [B, H//window_size, window_size, W//window_size, window_size, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [window_size, window_size]
        self.num_heads = num_heads
        head_dim = dim // num_heads  # the number of channels for every head
        self.scale = head_dim ** -0.5 or qk_scale  # the inverse of the square root of d

        # define a parameter table of relative position bias
        # Zero-all matrix, shape: (2*window_size-1) * (2*window_size-1), num_heads
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w))  # [2, Wh, Ww] , torch.meshgrid get grids
        coords_flatten = torch.flatten(coords, 1)  # [2, window_size*window_size]
        # [2, window_size*window_size, 1] - [2, 1, window_size*window_size]，下方仍然采用广播机制
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # [2, window_size*window_size, window_size*window_size]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [window_size*window_size,
        # window_size*window_size, 2]
        # 行列标加上M-1
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        # 行标加上2M-1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # 行列标相加
        relative_position_index = relative_coords.sum(-1)  # [window_size*window_size, window_size*window_size]
        # 保存相对位置索引，固定值，不需要再次修改
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 线性变换生成指定输出维度的tensor，bias=True，说明三个分量相同
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)  # timm库中的函数, 对bias查找表在指定标准差的情况下进行初始化
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, window_size*window_size, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Wh*Ww, total_embed_dim(C)]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, window_size*window_size, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, window_size*window_size, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, window_size*window_size, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #  上一步中, self.qkv(x)在dim(即channel)维度上进行对x进行线性变换, 输出shape: (num_windows*B, window_size*window_size, 3*C)
        # 接着进行reshape操作, 输出shape: (num_windows*B, window_size*window_size, 3, num_heads, C // self.num_heads)
        # 最后进行维度的前后调节, 输出shape: (3, num_windows*B, num_heads, window_size*window_size, C // self.num_heads)
        # 分成3份, query/key/value, 每份的shape为: (num_windows*B, num_heads, window_size*window_size, C // self.num_heads)
        # [batch_size*num_windows, num_heads, window_size*window_size, embed_dim_per_head]
        # q, k, v = qkv.unbind(0)  # make torch-script happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, window_size*window_size]
        # @: multiply -> [batch_size*num_windows, num_heads, window_size*window_size, window_size*window_size]
        q = q * self.scale  # 前乘后乘都可以, 套用公式: q/sqrt(d)
        # 对key的最后两个维度进行调换, 调换后的shape: (num_windows*B, num_heads, C // self.num_heads, window_size*window_size)
        # 然后计算q*(k转置), shape: (num_windows*B, num_heads, window_size*window_size, window_size*window_size)
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [window_size*window_size*window_size*window_size,
        # nH] -> [window_size*window_size,
        # window_size*window_size,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        # [num_windows*Batch_size, window_size*window_size, window_size*window_size]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        # 为attn中的每一个元素增加一个均值为0, 标准差为0.02的小扰动, 论文中说这个做法能够显著优化后续结果
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, window_size*window_size, window_size*window_size],nW=num_windows
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, window_size*window_size, window_size*window_size]
            # mask.unsqueeze: [1, nW, 1, window_size*window_size, window_size*window_size]
            # mask中只有0和负无穷两个取值, 当对应元素与0相加时,无影响, 当与负无穷相加时, 在后续计算softmax时该项无限接近于0, 起到了掩膜的作用
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            # attn = rearrange(attn, "b W H n n -> (b W) H n n", H=self.num_heads, n=N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)  # 套用公式, 在最后一个维度上进行softmax, 使其归一化至0~1之间
        attn = self.attn_drop(attn)

        # @: multiply,套用公式，与value相乘 -> [batch_size*num_windows, num_heads, window_size*window_size, embed_dim_per_head]
        # transpose: 按照指定维度调换顺序 -> [batch_size*num_windows, window_size*window_size, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, window_size*window_size, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # a*b 逐元素乘积， a@b 等同于 np.matmul 矩阵乘法
        x = self.proj(x)  # 进行一次全连接层, 输入输出的通道数均为C
        x = self.proj_drop(x)  # 并未进行实际的Dropout操作
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qkv_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # 隐藏层维度: 4*dim
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, attn_mask):
        """ forward function
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            attn_mask: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍,因为是在高度方向的下侧和宽度方向的右侧padding，故pad_left和pad_top=0
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))  # 与AttenMask的Padding相同
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, window_size, window_size, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, window_size*window_size, C]
        # x_windows = rearrange(x_windows, "b h w c -> b (h w) c")

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, window_size*window_size, C]

        # merge windows
        # [nW*B, window_size, window_size, C]
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # attn_windows = rearrange(attn_windows, "b (h w) c -> b h w c", h=self.window_size, w=self.window_size)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H, W, C]

        # reverse cyclic shift, 偶数层为加快自注意力计算速度曾将特征图向左上方shift了shift_size，
        # 此处需要将经过shift之后的自注意力结果向右下方shift, 起到还原的作用
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()  # 把前面pad的数据移除掉，取填补之前的有效范围

        x = x.view(B, H * W, C)  # 将特征图压缩为特征向量形式
        # FFN
        x = shortcut + self.drop_path(x)  # 残差边中, 原始特征先与drop_path(x)相加
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # 先进行LN, 再计算mlp
        # Mlp实际上就是一个三层的全连接, 第一层为输入层, 特征数量为dim;
        # 第二层为隐藏层, 特征数量为4 * dim;
        # 第三层为输出层, 特征数量又变为dim
        # Multi Layer Perception(MLP 多层感知机)的目的就是增加网络的深度
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"  # 若 L!= H*W,报错

        x = rearrange(x, "b (h w) c -> b h w c", h=H, w=W)  # equal to x = x.view(B, H, W, C)

        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding,使得padding之后的特征图宽高为2的倍数
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        # 四种patch均以2为间隔进行采样
        # 以0::2为例, 意义为以0为开始, 以列表终点为结束, 步长为2, 因此访问的都是偶数索引
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]，将以上四个子特征图在channel维度上进行concat, 当前channel数量为原来的4倍
        # x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]， 将concat后的特征图进行压扁, 由二维结构变为一维向量
        x = rearrange(x, "b h w c -> b (h w) c")

        x = self.norm(x)  # LN归一化
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]， 线性变换, 将之前concat后的特征向量的channel数由4倍变换为2倍

        return x  # 输出合并后的特征向量, 相邻的4个小token合并为一个大token, 通道数为原来的2倍


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qkv_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.depth = depth  # 当前stage的block数, 可能为2/2/6/2
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2  # 窗口移动量，论文中为H/2和 W/2,向下取整

        # build blocks, 按照给定的block数量进行当前stage构建
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qkv_scale=qkv_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        # 构建完当前stage后需要进行合并操作, 前三个stage为PatchMerging, 最后一个stage为None
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.
            Args:
                x: Input feature, tensor size (B, H*W, C).
                H, W: Spatial resolution of the input feature.
        """
        # calculate attention mask for SW-MSA
        # 按照window_size的整数倍向上取整
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size

        # 构建原始mask, 全零, shape: (1, Hp, Wp, 1)，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        # 切片设定与下方的赋值，实现相同数字对应连续区域
        h_slices = (slice(0, -self.window_size),  # [0, Hp-window_size)
                    slice(-self.window_size, -self.shift_size),  # [Hp-window_size, Hp-shift_size)
                    slice(-self.shift_size, None))  # [Hp-shift_size, Hp)
        w_slices = (slice(0, -self.window_size),  # 同h_slices
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        # Hp和Wp为4, 且window_size为2时的mask结果类似于:
        # 0, 0, 1, 2
        # 0, 0, 1, 2
        # 3, 3, 4, 5
        # 6, 6, 7, 8

        # 使用window_partition将img_mask划分为窗口
        mask_windows = window_partition(img_mask, self.window_size)  # [nW, window_size, window_size, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, window_size*window_size]
        # mask_windows = rearrange(mask_windows, "b h w c -> b (h w c)")
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # [nW, 1, window_size*window_size] - [nW, window_size*window_size, 1], 涉及到广播机制
        # [nW, window_size*window_size, window_size*window_size]，masked_fill 相同区域（值为0），填入0，不同区域填入-100（表示mask）
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        # 同一个Block中多次使用的SW-MSA的mask是相同的
        for blk in self.blocks:  # 遍历当前stage中的每一个SwinTransformerBlock
            blk.H, blk.W = H, W  # 传入token在宽高方向上的数量
            if not torch.jit.is_scripting() and self.use_checkpoint:  # 使用checkpoint可以减少内存消耗
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)  # 若无checkpoint, 则直接执行SwinTransformerBlock的forward函数

        if self.downsample is not None:  # 如果不是最后一个stage则进入
            x_down = self.downsample(x, H, W)  # 利用PatchMerging进行降采样(即合并)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2  # 降采样后的特征图宽高
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)  # timm库中的函数, 由单个数字构建包含两个对象的元组, 即每个patch中相应的行像素数和列像素数
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        # 利用卷积实现图像块切分和线性映射, 卷积核的尺寸为patch的尺寸, 卷积过程中的跨度为patch的尺寸, 能够减小输出特征图的尺寸,
        # 实现了对原图进行切分的功能, 避免了patch之间在线性映射过程中的重叠
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape  # 获取原始图像的宽高

        # padding
        # 对原始图像进行pad补全操作, 使其宽高均为patch_size的整数倍, 默认补0
        # to pad the last 3 dimensions, (W_left, W_right, H_top, H_bottom, C_front, C_back)
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        # 对x利用卷积的方式进行切分和线性映射, 使其通道数由3变为设置的96, 以token为最小单位, x的尺寸=原始尺寸/patch_size
        x = self.proj(x)  # B C Wh Ww

        # self.norm set None as default, 以下不执行
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            # flatten: [B, C, H, W] -> [B, C, HW]
            # transpose: [B, C, HW] -> [B, HW, C]
            # x = x.flatten(2).transpose(1, 2)
            x = rearrange(x, "n c h w -> n (h w) c")  # eniops 替代写法
            x = self.norm(x)
            # x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
            x = rearrange(x, " n (h w) c -> n c h w", c=self.embed_dim, h=Wh, w=Ww)
        return x


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input images channels. Default: 3
        embed_dim (int): Patch embedding dimension,as C in paper. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate,the first one used in the pos Drop、MLP and others. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.2
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode). -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=224, patch_size=4, in_chans=3, embed_dim=96, depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4., qkv_bias=True, qkv_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=nn.LayerNorm, ape=False,
                 patch_norm=True, out_indices=(0, 1, 2, 3), frozen_stages=-1, use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split images into non-overlapping patches, patch Partition and Linear Embedding
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 注意这里构建的stage和论文图中有些差异
            # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的
            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),  # stage1:96;stage2:192;stage3:384;stage4:768
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                qkv_scale=qkv_scale,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                # 实质上代码中将下一stage的PatchMerging放到当前stage中实现,相当于下采样
                                use_checkpoint=use_checkpoint)
            self.layers.append(layers)

        # stage4输出特征矩阵的channels, embed_dim * 8
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def _init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pretrained weights.
            Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        # 一. 图像块切分与线性变换 x: [B, L, C]
        x = self.patch_embed(x)

        # 二. 特征图展平 & Dropout防止过拟合(未使用Dropout功能)
        Wh, Ww = x.size(2), x.size(3)  # 特征图中token的横竖个数
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
            # x的维度为[b,c,Wh,Ww], 将Wh和Ww维度展平为一个维度, 得到[b,c,Wh*Ww], 然后对维度进行调换, 得到[b, Wh*Ww,c]
        x = self.pos_drop(x)

        outs = list()  # 待输出的四个stage的特征图列表
        # stage1, stage2, stage3, stage4
        for i in range(self.num_layers):  # num_layer = 4
            layer = self.layers[i]  # 对应着各个stage, 一个layer就是一个BasicLayer
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)  # 每个stage的最终产物, x_out为当前stage生成的特征图, x为下一个stage输入的特征图
            if i in self.out_indices:  # out_indices: (0, 1, 2, 3)  需要存储的指定stage产物
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)  # 对各stage输出特征图先进行LN归一化
                # out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                out = rearrange(x_out, " b (h w) c -> b c h w", c=self.num_features[i], h=H, w=W)
                out = out.contiguous()
                outs.append(out)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()


params = dict(
    tiny={
        # [96, 192, 384, 768]
        # trained ImageNet-1K, official pretrain weights:
        # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
        "window_size": 7, "embed_dim": 96, "depths": (2, 2, 6, 2), "num_heads": (3, 6, 12, 24),
        "ape": False, "drop_path_rate": 0.3, "patch_norm": True, "use_checkpoint": False
    },
    small={
        # [96, 192, 384, 768]
        # trained ImageNet-1K, official pretrain weights:
        # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
        "window_size": 7, "embed_dim": 96, "depths": (2, 2, 18, 2), "num_heads": (3, 6, 12, 24),
        "ape": False, "drop_path_rate": 0.3, "patch_norm": True, "use_checkpoint": False
    },
    base={
        # [128, 256, 512, 1024], pretrain_image_size: 224 (optional, 384)
        # trained ImageNet-1K,official pretrain weights:
        # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth
        "window_size": 7, "embed_dim": 128, "depths": (2, 2, 18, 2), "num_heads": (4, 8, 16, 32),
        "ape": False, "drop_path_rate": 0.3, "patch_norm": True, "use_checkpoint": False
    },
    large={
        # [192, 384,768, 1536]
        # trained ImageNet-22K, official pretrain weights:
        # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth
        "window_size": 7, "embed_dim": 192, "depths": (2, 2, 18, 2), "num_heads": (6, 12, 24, 48),
        "ape": False, "drop_path_rate": 0.3, "patch_norm": True, "use_checkpoint": False
    },
    # xlarge={  # [192, 384,768, 1536]
    #     "window_size": 12, "embed_dim": 192, "depths": (2, 2, 18, 2), "num_heads": (6, 12, 24, 48),
    #     "ape": False, "drop_path_rate": 0.3, "patch_norm": True, "use_checkpoint": False
    # }
)


def SwinNet(in_chans, Swin_type: str):
    param = params[Swin_type]
    encoder = SwinTransformer(in_chans=in_chans, embed_dim=param["embed_dim"],
                              depths=param["depths"], num_heads=param["num_heads"],
                              window_size=param["window_size"], ape=param["ape"],
                              drop_path_rate=param["drop_path_rate"], patch_norm=param["patch_norm"],
                              use_checkpoint=param["use_checkpoint"])
    return encoder


if __name__ == '__main__':
    data = torch.ones((4, 3, 512, 512))
    # backbone = swin_tiny_patch4_window7_224(num_classes=2)
    model = SwinNet(3, "base")
    with torch.no_grad():
        out = model(data)
    print(out)
