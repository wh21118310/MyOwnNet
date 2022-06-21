import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.backbone.Res2Net import res2net50_v1b_26w_4s
from nets.backbone.bk import Backbone


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class NeighborConnectionDecoder(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(x2_1)) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


# Group-Reversal Attention (GRA) Block
class GRA(nn.Module):
    def __init__(self, channel, subchannel):
        super(GRA, self).__init__()
        self.group = channel // subchannel
        self.conv = nn.Sequential(
            nn.Conv2d(channel + self.group, channel, 3, padding=1), nn.ReLU(True),
        )
        self.score = nn.Conv2d(channel, 1, 3, padding=1)

    def forward(self, x, y):
        if self.group == 1:
            x_cat = torch.cat((x, y), 1)
        elif self.group == 2:
            xs = torch.chunk(x, 2, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y), 1)
        elif self.group == 4:
            xs = torch.chunk(x, 4, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y), 1)
        elif self.group == 8:
            xs = torch.chunk(x, 8, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y), 1)
        elif self.group == 16:
            xs = torch.chunk(x, 16, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
                               xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y), 1)
        elif self.group == 32:
            xs = torch.chunk(x, 32, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
                               xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y,
                               xs[16], y, xs[17], y, xs[18], y, xs[19], y, xs[20], y, xs[21], y, xs[22], y, xs[23], y,
                               xs[24], y, xs[25], y, xs[26], y, xs[27], y, xs[28], y, xs[29], y, xs[30], y, xs[31], y),
                              1)
        else:
            raise Exception("Invalid Channel")

        x = x + self.conv(x_cat)
        y = y + self.score(x)

        return x, y


class ReverseStage(nn.Module):
    def __init__(self, channel):
        super(ReverseStage, self).__init__()
        self.weak_gra = GRA(channel, channel)
        self.medium_gra = GRA(channel, 8)
        self.strong_gra = GRA(channel, 1)

    def forward(self, x, y):
        # reverse guided block
        y = 1 - (1 * (torch.sigmoid(y)))

        # three group-reversal attention blocks
        x, y = self.weak_gra(x, y)
        x, y = self.medium_gra(x, y)
        _, y = self.strong_gra(x, y)

        return y


backbone_names1 = [  # C: 2048,1024,512,256
    'resnet50', 'res2net50',
]
backbone_names2 = [  # C: 1024,512,256,128
    'convnext_base', 'swinT_base'
]


class SearchIdentificationNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32, pretrained=False, bk='res2net50'):
        super(SearchIdentificationNet, self).__init__()
        # ----model Name ------
        self.model = bk
        # ---- ResNet Backbone ----
        self.backbone = Backbone(bk=bk, pretrain=pretrained, in_channels=3)
        # ---- Receptive Field Block like module ----
        if bk in backbone_names1:
            self.rfb2_1 = RFB_modified(512, channel)
            self.rfb3_1 = RFB_modified(1024, channel)
            self.rfb4_1 = RFB_modified(2048, channel)
        elif bk in backbone_names2:
            self.rfb2_1 = RFB_modified(256, channel)
            self.rfb3_1 = RFB_modified(512, channel)
            self.rfb4_1 = RFB_modified(1024, channel)
        # ---- Partial Decoder ----
        self.NCD = NeighborConnectionDecoder(channel)

        # # ---- reverse stage ----
        self.RS5 = ReverseStage(channel)
        self.RS4 = ReverseStage(channel)
        self.RS3 = ReverseStage(channel)

    def forward(self, x):
        # Feature Extraction
        _, x2, x3, x4 = self.backbone(x, self.model)

        # Receptive Field Block (enhanced)
        x2_rfb = self.rfb2_1(x2)  # channel -> 32
        x3_rfb = self.rfb3_1(x3)  # channel -> 32
        x4_rfb = self.rfb4_1(x4)  # channel -> 32

        # Neighbourhood Connected Decoder
        S_g = self.NCD(x4_rfb, x3_rfb, x2_rfb)

        # ---- reverse stage 5 ----
        guidance_g = F.interpolate(S_g, scale_factor=0.25, mode='bilinear')
        guidance_g = torch.relu(guidance_g)
        ra4_feat = self.RS5(x4_rfb, guidance_g)
        S_5 = ra4_feat + guidance_g

        # ---- reverse stage 4 ----
        guidance_5 = F.interpolate(S_5, scale_factor=2, mode='bilinear')
        guidance_5 = torch.relu(guidance_5)
        ra3_feat = self.RS4(x3_rfb, guidance_5)
        S_4 = ra3_feat + guidance_5

        # ---- reverse stage 3 ----
        guidance_4 = F.interpolate(S_4, scale_factor=2, mode='bilinear')
        guidance_4 = torch.relu(guidance_4)
        ra2_feat = self.RS3(x2_rfb, guidance_4)
        S_3 = ra2_feat + guidance_4

        # S_g_pred = F.interpolate(S_g, scale_factor=8, mode='bilinear')  # Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
        # S_5_pred = F.interpolate(S_5, scale_factor=32, mode='bilinear')  # Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)
        # S_4_pred = F.interpolate(S_4, scale_factor=16, mode='bilinear')  # Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)
        S_3_pred = F.interpolate(S_3, scale_factor=8, mode='bilinear')  # Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
        if self.model == 'convnext_base':
            # S_g_pred = F.interpolate(S_g_pred, scale_factor=0.5, mode='bilinear')
            # S_5_pred = F.interpolate(S_5_pred, scale_factor=0.5, mode='bilinear')
            # S_4_pred = F.interpolate(S_4_pred, scale_factor=0.5, mode='bilinear')
            S_3_pred = F.interpolate(S_3_pred, scale_factor=0.5, mode='bilinear')
        # return S_g_pred, S_5_pred, S_4_pred, S_3_pred
        return S_3_pred


from .backbone.Pyramid_Vision_Transformer import pvt_tiny, pvt_medium, pvt_large


class SearchIdentificationNet_PVT(nn.Module):
    def __init__(self, bk='large', channel=32, img_size=224):
        super(SearchIdentificationNet_PVT, self).__init__()
        if bk == 'tiny':
            self.backbone = pvt_tiny(inchans=3, F4=False, img_size=img_size)
        elif bk == 'medium':
            self.backbone = pvt_medium(inchans=3, F4=False, img_size=img_size)
        elif bk == 'large':
            self.backbone = pvt_large(inchans=3, F4=False, img_size=img_size)
        self.rfb2_1 = RFB_modified(128, channel)
        self.rfb3_1 = RFB_modified(320, channel)
        self.rfb4_1 = RFB_modified(512, channel)
        # ---- Partial Decoder ----
        self.NCD = NeighborConnectionDecoder(channel)
        # ---- reverse stage ----
        self.RS5 = ReverseStage(channel)
        self.RS4 = ReverseStage(channel)
        self.RS3 = ReverseStage(channel)

        self.UpSample_g = nn.Upsample(scale_factor=0.25, mode='bilinear')
        self.UpSample_5 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.UpSample_4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.UpSample_Sg = nn.Upsample(scale_factor=8, mode='bilinear')
        self.UpSample_S5 = nn.Upsample(scale_factor=32, mode='bilinear')
        self.UpSample_S4 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.UpSample_S3 = nn.Upsample(scale_factor=8, mode="bilinear")

    def forward(self, x):
        # Feature Extraction
        _, x2, x3, x4 = self.backbone(x)

        # Receptive Field Block (enhanced)
        x2_rfb = self.rfb2_1(x2)  # channel -> 64
        x3_rfb = self.rfb3_1(x3)  # channel -> 64
        x4_rfb = self.rfb4_1(x4)  # channel -> 64

        # Neighbourhood Connected Decoder
        S_g = self.NCD(x4_rfb, x3_rfb, x2_rfb)

        # ---- reverse stage 5 ----
        guidance_g = self.UpSample_g(S_g)
        guidance_g = torch.relu(guidance_g)
        ra4_feat = self.RS5(x4_rfb, guidance_g)
        S_5 = ra4_feat + guidance_g

        # ---- reverse stage 4 ----
        guidance_5 = self.UpSample_5(S_5)
        guidance_5 = torch.relu(guidance_5)
        ra3_feat = self.RS4(x3_rfb, guidance_5)
        S_4 = ra3_feat + guidance_5

        # ---- reverse stage 3 ----
        guidance_4 = self.UpSample_4(S_4)
        guidance_4 = torch.relu(guidance_4)
        ra2_feat = self.RS3(x2_rfb, guidance_4)
        S_3 = ra2_feat + guidance_4

        S_g_pred = self.UpSample_Sg(S_g)
        S_5_pred = self.UpSample_S5(S_5)
        S_4_pred = self.UpSample_S4(S_4)
        S_3_pred = self.UpSample_S3(S_3)  # Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
        # return S_g_pred, S_5_pred, S_4_pred, S_3_pred
        return S_3_pred


if __name__ == '__main__':
    import numpy as np
    from time import time
    net = SearchIdentificationNet_PVT(bk='large')
    # net = SearchIdentificationNet(bk='convnext_base')
    net.eval()

    dump_x = torch.randn(1, 3, 512, 512)
    frame_rate = np.zeros((1000, 1))
    for i in range(1):
        start = time()
        y = net(dump_x)
        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        print(i, '->', running_frame_rate)
        frame_rate[i] = running_frame_rate
    print(np.mean(frame_rate))
    print(y.shape)
