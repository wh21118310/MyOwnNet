"""
 @Time    : 2021/7/6 14:23
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2021_PFNet
 @File    : PFNet.py
 @Function: Focus and Exploration Network
 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchsummary import summary
###################################################################
# ################## Backbone ######################
###################################################################
from nets.backbone.Swin_transformer import SwinNet
from nets.backbone.convnext import ConvNeXt_Seg
from nets.backbone.resnet import resnet50

models_type1 = [
            'resnet50', 'convnext_base',
        ]


class Backbone(nn.Module):
    def __init__(self, bk="resnet50", pretrain=False, in_channels=3):
        super(Backbone, self).__init__()
        self.backbone = bk
        if self.backbone == 'resnet50':
            self.backbone = resnet50(pretrain)
            self.layer1 = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu,
                                        self.backbone.maxpool, self.backbone.layer1)
            self.layer2 = self.backbone.layer2
            self.layer3 = self.backbone.layer3
            self.layer4 = self.backbone.layer4
        elif self.backbone == 'convnext_base':
            self.backbone = ConvNeXt_Seg(in_channels, model_type="base", pretrained=pretrain)
            self.downsample = self.backbone.downsample_layers
            self.stage = self.backbone.stages
            self.layer1 = nn.Sequential(self.downsample[0], self.stage[0], self.backbone.norm0)
            self.layer2 = nn.Sequential(self.downsample[1], self.stage[1], self.backbone.norm1)
            self.layer3 = nn.Sequential(self.downsample[2], self.stage[2], self.backbone.norm2)
            self.layer4 = nn.Sequential(self.downsample[3], self.stage[3], self.backbone.norm3)
        elif self.backbone == 'swinT_base':
            self.backbone = SwinNet(3, "base")
            self.pos_drop = self.backbone.pos_drop
            self.layers = self.backbone.layers
            self.final_norm = self.backbone.norm
            # self.layer1 = self.layers[0]
            self.layer1 = self.backbone.patch_embed
            self.layer2 = self.layers[0]
            self.layer3 = self.layers[1]
            self.layer4_1 = self.layers[2]
            self.layer4_2 = self.layers[3]

    def forward(self, x, model_type='resnet50'):
        if model_type in models_type1:
            layer1 = self.layer1(x)  # [-1, 256, h/4, w/4]
            layer2 = self.layer2(layer1)  # [-1, 512, h/8, w/8]
            layer3 = self.layer3(layer2)  # [-1, 1024, h/16, w/16]
            layer4 = self.layer4(layer3)  # [-1, 2048, h/32, w/32]
            return layer1, layer2, layer3, layer4
        elif model_type == "swinT_base":
            x, H, W = self.layer1(x)
            x = self.pos_drop(x)
            layer1 = rearrange(x, " b (h w) c-> b c h w", h=H, w=W)
            x, H, W = self.layer2(x, H, W)
            layer2 = rearrange(x, " b (h w) c-> b c h w", h=H, w=W)
            x, H, W = self.layer3(x, H, W)
            layer3 = rearrange(x, " b (h w) c-> b c h w", h=H, w=W)
            x, H, W = self.layer4_1(x, H, W)
            x, H, W = self.layer4_2(x, H, W)
            layer4 = rearrange(x, " b (h w) c-> b c h w", h=H, w=W)
            return layer1, layer2, layer3, layer4


###################################################################
# ################## Channel Attention Block ######################
###################################################################
class CA_Block(nn.Module):
    def __init__(self, in_dim):
        super(CA_Block, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : channel attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


###################################################################
# ################## Spatial Attention Block ######################
###################################################################
class SA_Block(nn.Module):
    def __init__(self, in_dim):
        super(SA_Block, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : spatial attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


###################################################################
# ################## Context Exploration Block ####################
###################################################################
class Context_Exploration_Block(nn.Module):
    def __init__(self, input_channels):
        super(Context_Exploration_Block, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)

        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p2 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p3 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p4 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
                                    nn.BatchNorm2d(self.input_channels), nn.ReLU())

    def forward(self, x):
        p1_input = self.p1_channel_reduction(x)
        p1 = self.p1(p1_input)
        p1_dc = self.p1_dc(p1)

        p2_input = self.p2_channel_reduction(x) + p1_dc
        p2 = self.p2(p2_input)
        p2_dc = self.p2_dc(p2)

        p3_input = self.p3_channel_reduction(x) + p2_dc
        p3 = self.p3(p3_input)
        p3_dc = self.p3_dc(p3)

        p4_input = self.p4_channel_reduction(x) + p3_dc
        p4 = self.p4(p4_input)
        p4_dc = self.p4_dc(p4)

        ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))

        return ce


###################################################################
# ##################### Positioning Module ########################
###################################################################
class Positioning(nn.Module):
    def __init__(self, channel):
        super(Positioning, self).__init__()
        self.channel = channel
        self.cab = CA_Block(self.channel)
        self.sab = SA_Block(self.channel)
        self.map = nn.Conv2d(self.channel, 1, 7, 1, 3)

    def forward(self, x):
        cab = self.cab(x)
        sab = self.sab(cab)
        map = self.map(sab)

        return sab, map


###################################################################
# ######################## Focus Module ###########################
###################################################################
class Focus(nn.Module):
    def __init__(self, channel1, channel2):
        super(Focus, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
                                nn.BatchNorm2d(self.channel1), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))

        self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())
        self.output_map = nn.Conv2d(self.channel1, 1, 7, 1, 3)
        self.fp = Context_Exploration_Block(self.channel1)
        self.fn = Context_Exploration_Block(self.channel1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.bn1 = nn.BatchNorm2d(self.channel1)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(self.channel1)
        self.relu2 = nn.ReLU()

    def forward(self, x, y, in_map):
        # x; current-level features
        # y: higher-level features
        # in_map: higher-level prediction

        up = self.up(y)

        input_map = self.input_map(in_map)
        f_feature = x * input_map
        b_feature = x * (1 - input_map)

        fp = self.fp(f_feature)
        fn = self.fn(b_feature)

        refine1 = up - (self.alpha * fp)
        refine1 = self.bn1(refine1)
        refine1 = self.relu1(refine1)

        refine2 = refine1 + (self.beta * fn)
        refine2 = self.bn2(refine2)
        refine2 = self.relu2(refine2)

        output_map = self.output_map(refine2)

        return refine2, output_map


###################################################################
# ########################## NETWORK ##############################
###################################################################
class PFNet(nn.Module):
    def __init__(self, pretrain=False, bk='resnet50'):
        super(PFNet, self).__init__()
        # params
        self.model = bk
        # backbone
        self.backbone = Backbone(bk=bk, pretrain=pretrain)

        # channel reduction
        if bk == 'resnet50':
            self.cr4 = nn.Sequential(nn.Conv2d(2048, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU())
            self.cr3 = nn.Sequential(nn.Conv2d(1024, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
            self.cr2 = nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU())
            self.cr1 = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        elif bk in ['convnext_base', 'swinT_base']:
            self.cr4 = nn.Sequential(nn.Conv2d(1024, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
            self.cr3 = nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU())
            self.cr2 = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
            self.cr1 = nn.Sequential(nn.Conv2d(128, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU())

        # positioning
        if bk == 'resnet50':
            self.positioning = Positioning(512)
        elif bk in ['convnext_base', 'swinT_base']:
            self.positioning = Positioning(256)
        # focus
        if bk == 'resnet50':
            self.focus3 = Focus(256, 512)
            self.focus2 = Focus(128, 256)
            self.focus1 = Focus(64, 128)
        elif bk in ['convnext_base', 'swinT_base']:
            self.focus3 = Focus(128, 256)
            self.focus2 = Focus(64, 128)
            self.focus1 = Focus(32, 64)
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        # x: [batch_size, channel=3, h, w]
        layer1, layer2, layer3, layer4 = self.backbone(x, self.model)

        # channel reduction
        cr4 = self.cr4(layer4)
        cr3 = self.cr3(layer3)
        cr2 = self.cr2(layer2)
        cr1 = self.cr1(layer1)

        # positioning
        positioning, predict4 = self.positioning(cr4)

        # focus
        focus3, predict3 = self.focus3(cr3, positioning, predict4)
        focus2, predict2 = self.focus2(cr2, focus3, predict3)
        focus1, predict1 = self.focus1(cr1, focus2, predict2)

        # rescale
        # predict4 = F.interpolate(predict4, size=x.size()[2:], mode='bilinear', align_corners=True)
        # predict3 = F.interpolate(predict3, size=x.size()[2:], mode='bilinear', align_corners=True)
        # predict2 = F.interpolate(predict2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(predict1, size=x.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            # return predict4, predict3, predict2, predict1
            return predict1
        # return torch.sigmoid(predict4), torch.sigmoid(predict3), torch.sigmoid(predict2), torch.sigmoid(
        #     predict1)
        return torch.sigmoid(predict1)


if __name__ == '__main__':
    data = torch.rand((4, 3, 512, 512)).cuda()
    net = PFNet(bk='swinT_base').cuda()
    # net = PFNet()
    result = net(data)
    print(result)
