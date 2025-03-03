import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
sys.path.append("..")
from utils import local_pcd
from .homography import inverse_warping

eps = 1e-12

def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return

class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)



class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)



class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, pad=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class Hourglass3d(nn.Module):
    def __init__(self, channels):
        super(Hourglass3d, self).__init__()

        self.conv1a = ConvBnReLU3D(channels, channels * 2, kernel_size=3, stride=2, pad=1)
        self.conv1b = ConvBnReLU3D(channels * 2, channels * 2, kernel_size=3, stride=1, pad=1)

        self.conv2a = ConvBnReLU3D(channels * 2, channels * 4, kernel_size=3, stride=2, pad=1)
        self.conv2b = ConvBnReLU3D(channels * 4, channels * 4, kernel_size=3, stride=1, pad=1)

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose3d(channels * 4, channels * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels * 2))

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose3d(channels * 2, channels, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels))

        self.redir1 = ConvBn3D(channels, channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = ConvBn3D(channels * 2, channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1b(self.conv1a(x))
        conv2 = self.conv2b(self.conv2a(conv1))
        dconv2 = F.relu(self.dconv2(conv2) + self.redir2(conv1), inplace=True)
        dconv1 = F.relu(self.dconv1(dconv2) + self.redir1(x), inplace=True)
        return dconv1


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea

class DeConv2dFuse(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(DeConv2dFuse, self).__init__()

        self.deconv = Deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                               bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2d(2*out_channels, out_channels, kernel_size, stride=1, padding=1,
                           bn=bn, relu=relu, bn_momentum=bn_momentum)

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x


class FeatureNet(nn.Module):
    def __init__(self, base_channels, num_stage=3, stride=4, arch_mode="fpn"):
        super(FeatureNet, self).__init__()
        assert arch_mode in ["unet", "fpn"], print("mode must be in 'unet' or 'fpn', but get:{}".format(arch_mode))
        print("*************feature extraction arch mode:{}****************".format(arch_mode))
        self.arch_mode = arch_mode
        self.stride = stride
        self.base_channels = base_channels
        self.num_stage = num_stage

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.out_channels = [4 * base_channels]

        if self.arch_mode == 'unet':
            if num_stage == 3:
                self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)
                self.deconv2 = DeConv2dFuse(base_channels * 2, base_channels, 3)

                self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
                self.out3 = nn.Conv2d(base_channels, base_channels, 1, bias=False)
                self.out_channels.append(2 * base_channels)
                self.out_channels.append(base_channels)

            elif num_stage == 2:
                self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)

                self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
                self.out_channels.append(2 * base_channels)
        elif self.arch_mode == "fpn":
            final_chs = base_channels * 4
            if num_stage == 3:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
                self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
                self.out3 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels * 2)
                self.out_channels.append(base_channels)

            elif num_stage == 2:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        # print("************* the intra_feat of origin is :{}****************".format(intra_feat))
        outputs = {}
        out = self.out1(intra_feat)
        # print("&&&&&&&&& the intra_feat of operate is :{}&&&&&&&&".format(intra_feat))
        # print("********************+++++++++++++")
        # print("the shape of stage1 out is: ", out.shape)   # torch.size([8,32,128,160])
        outputs["stage1"] = out
        if self.arch_mode == "unet":
            if self.num_stage == 3:
                intra_feat = self.deconv1(conv1, intra_feat)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

                intra_feat = self.deconv2(conv0, intra_feat)
                out = self.out3(intra_feat)
                outputs["stage3"] = out

            elif self.num_stage == 2:
                intra_feat = self.deconv1(conv1, intra_feat)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

        elif self.arch_mode == "fpn":
            if self.num_stage == 3:
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
                out = self.out2(intra_feat)
                # print("the shape of stage2 out is: ", out.shape)  # torch.size([8,16,256,320])
                outputs["stage2"] = out

                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
                out = self.out3(intra_feat)
                # print("the shape of stage3 out is: ", out.shape)  # torch.size([8,8,512,640])
                outputs["stage3"] = out

            elif self.num_stage == 2:
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

        return outputs

# Geo-MVSNet(4阶段)所用的cost volume reg
class Reg2d(nn.Module):
    def __init__(self, input_channel=32, base_channel=8):
        super(Reg2d, self).__init__()
        
        self.conv0 = ConvBnReLU3D(input_channel, base_channel, kernel_size=(1,3,3), pad=(0,1,1))
        self.conv1 = ConvBnReLU3D(base_channel, base_channel*2, kernel_size=(1,3,3), stride=(1,2,2), pad=(0,1,1))
        self.conv2 = ConvBnReLU3D(base_channel*2, base_channel*2)

        self.conv3 = ConvBnReLU3D(base_channel*2, base_channel*4, kernel_size=(1,3,3), stride=(1,2,2), pad=(0,1,1))
        self.conv4 = ConvBnReLU3D(base_channel*4, base_channel*4)

        self.conv5 = ConvBnReLU3D(base_channel*4, base_channel*8, kernel_size=(1,3,3), stride=(1,2,2), pad=(0,1,1))
        self.conv6 = ConvBnReLU3D(base_channel*8, base_channel*8)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*8, base_channel*4, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(base_channel*4),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*4, base_channel*2, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(base_channel*2),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*2, base_channel, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(base_channel),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 1, stride=1, padding=0)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)

        return x.squeeze(1)


# CasMVSNet的正则化过程
class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(CostRegNet, self).__init__()
        self.conv0 = Conv3d(in_channels, base_channels, padding=1)

        self.conv1 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=2, padding=1)
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1)

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1)

        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)

        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)

        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x

# adopt from Uni-MVSNet：可见性自适应权重网络
class AggWeightNetVolume(nn.Module):
    def __init__(self, in_channels=32):
        super(AggWeightNetVolume, self).__init__()
        self.conv0 = Conv3d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.w_net = nn.Sequential(
            Conv3d(in_channels, 1, kernel_size=1, stride=1, padding=0),
            Conv3d(1, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        """
        :param x: (b, c, d, h, w)
        :return: (b, 1, d, h, w)
        """
        # stemp = x.clone()
        # out = self.conv0(stemp)
        # print("the shape of out is: ", out.shape)  # torch.Size([2, 1, 64, 128, 160])
        w = self.w_net(x)
        
        return w

# for me: 自己参考AA-RMVSNet重新设计了自适应权重网络
# 但模型实际运行后的结果并不比之前的要好。
class AggWeightNetVolume2(nn.Module):
    def __init__(self, in_channels=32):
        super(AggWeightNetVolume2, self).__init__()
        self.conv0 = Conv3d(in_channels, 1, kernel_size=3, stride=1, padding=1)  # 将out_channel由4改为1
        self.ResnetBlock = nn.Sequential(
            Conv3d(1, 1, kernel_size=1, stride=1, padding=0),
            Conv3d(1, 1, kernel_size=1, stride=1, padding=0),
        )
        self.conv1 = Conv3d(1, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
         """
         :param x: (b, c, d, h, w)
         :return: (b, 1, d, h, w)
         """
         stem = self.conv0(x)
         # torch.Size([2, 1, 64, 128, 160]) torch.Size([2, 1, 32, 256, 320]) torch.Size([2, 1, 8, 512, 640])
        #  print("the shape of stem is: ", stem.shape)  
         out = self.ResnetBlock(stem) + stem
         # torch.Size([2, 1, 64, 128, 160]) torch.Size([2, 1, 32, 256, 320]) torch.Size([2, 1, 8, 512, 640])
        #  print("the shape of out is: ", out.shape)  
         w = self.conv1(out)
         # torch.Size([2, 1, 64, 128, 160]) torch.Size([2, 1, 32, 256, 320]) torch.Size([2, 1, 8, 512, 640])
        #  print("the shape of w is: ", w.shape)  
         return w


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
        concat = F.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined


def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        # print("regression dim <= 2")
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)

    return depth

# 有监督的, 用以损失不平衡，放大细节流
def compute_reconstr_loss(warped, ref, mask):
    photo_loss = F.smooth_l1_loss(warped*mask, ref*mask, reduction='mean')
    return photo_loss

# code by myself
# input: 从重建网路估计出来的深度值inputs、src和ref图像(B,N,C,H,W)、相机参数cams（B,N,2,4,4）、图像深度depth_GT
def cross_view_loss(inputs, imgs, sample_cams, depth_gt_ms, depth_loss_weights):
    # def forward(self, inputs, imgs, sample_cams, num_views=5, **kwargs):
    # def unsup_loss(inputs, imgs, sample_cams, num_views=5, **kwargs):

    # imgs: [B,N,C,H,W] -> 列表：有N个元祖[B,C,H,W]
    img_nums = torch.unbind(imgs, 1)
    num_views = len(img_nums)
    
    # depth_loss_weights = kwargs.get("dlossw", None)

    total_photo_loss = torch.tensor(0.0, dtype=torch.float32, device=inputs['stage1']["depth"].device,
                                    requires_grad=False)
    reconstr_loss = torch.tensor(0.0, dtype=torch.float32, device=inputs['stage1']["depth"].device,
                                    requires_grad=False)
    reconstr_photo_loss = torch.tensor(0.0, dtype=torch.float32, device=inputs['stage1']["depth"].device,
                                    requires_grad=False)
    
    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"].unsqueeze(1)  # b,1,h,w
        depth_gt = depth_gt_ms[stage_key].unsqueeze(1)  # （b,h,w）-> (b,1,h,w)

        ref_img = imgs[:, 0]  # b,c,h,w
        scale = depth_est.shape[-1] / ref_img.shape[-1]
        # ref_img = F.interpolate(ref_img, scale_factor=scale, mode='bilinear', align_corners=True)
        # ref_img = ref_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
        ref_cam = sample_cams[stage_key][:, 0]  # b,2,4,4   ref_img的相机参数

        reprojection_losses = []

        for view in range(1, num_views): # N-1个src_img
            view_img = imgs[:, view]  # 依次取出src_img进行后续的图像合成
            view_cam = sample_cams[stage_key][:, view]   # src_img的相机参数
            view_img = F.interpolate(view_img, scale_factor=scale, mode='bilinear', align_corners=True)
            view_img = view_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
            # warp view_img to the ref_img using the dmap of the ref_img(depth_est、depth_gt)
            warped_img_est, mask_est = inverse_warping(view_img, ref_cam, view_cam, depth_est)
            warped_img_gt, mask_gt = inverse_warping(view_img, ref_cam, view_cam, depth_gt)
            # print("the shape of mask_est is: ", mask_est.shape)  #torch.Size([6, 128, 160, 1]) 
            # print(mask_est)  # mask_est的值不是0就是1
            mask = mask_est * mask_gt

            reconstr_loss = compute_reconstr_loss(warped_img_est, warped_img_gt, mask)
            # print("the shape of reconstr_loss is: ", reconstr_loss.shape)  # torch.size([]) 
            valid_mask = 1 - mask  # replace all 0 values with INF
            # 对整个图像进行处理，这样多张图像好进行比较，否则进行比较的时候回因为重建出来的mask区域不同而导致不规则的比较
            reprojection_losses.append(reconstr_loss + 1e4 * valid_mask)  # 对于非domain区域用INF值代替，这样后面的topk时候自然

        # (B,H,W,1) -> (N-1,B,H,W,1) -> (B,H,W,1,N-1)
        # top-k operates along the last dimension, so swap the axes accordingly
        reprojection_volume = torch.stack(reprojection_losses).permute(1, 2, 3, 4, 0)  # [6, 128, 160, 1, 2]
        # print('reprojection_volume: {}'.format(reprojection_volume.shape))
        # by default, it'll return top-k largest entries, hence sorted=False to get smallest entries
        top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=2, sorted=False)
        top_vals = torch.neg(top_vals)
        # print('top_vals: {}'.format(top_vals.shape))  # torch.Size([6, 128, 160, 1, 2]) 
        # top_mask = top_vals < (1e4 * torch.ones_like(top_vals, device=device))
        top_mask = top_vals < (1e4 * torch.ones_like(top_vals).cuda())
        top_mask = top_mask.float()
        # print("the shape of top_mask is: ", top_mask.shape)  # #torch.Size([6, 128, 160, 1, 2]) 
        # print(top_mask)  # 其值不是0就是1
        top_vals = torch.mul(top_vals, top_mask)  

        reconstr_photo_loss = torch.mean(torch.sum(top_vals, dim=-1))  #torch.size([6,128,160,1])
        stage_idx = int(stage_key.replace("stage", "")) - 1
        total_photo_loss += reconstr_photo_loss * depth_loss_weights[stage_idx]  # [0.5,1,2] as same as depth_l1_loss
        
        # 返回三阶段总的图像合成损失
    return total_photo_loss


# from me: 加入图像合成损失
def cas_mvsnet_loss(inputs, imgs, cam_para, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)

    total_depth_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    total_cpc_loss = cross_view_loss(inputs, imgs, cam_para, depth_gt_ms, depth_loss_weights)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_depth_loss += depth_loss_weights[stage_idx] * depth_loss
        else:
            total_depth_loss += 1.0 * depth_loss

    # L1 + Cross_View(图像合成损失的权重设置) 12、120、200、1
    total_loss = total_depth_loss + total_cpc_loss *12

    return total_loss, depth_loss, total_cpc_loss


# 未加入图像合成损失
# def cas_mvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
#     depth_loss_weights = kwargs.get("dlossw", None)

#     total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

#     for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
#         depth_est = stage_inputs["depth"]
#         depth_gt = depth_gt_ms[stage_key]
#         mask = mask_ms[stage_key]
#         mask = mask > 0.5

#         depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

#         if depth_loss_weights is not None:
#             stage_idx = int(stage_key.replace("stage", "")) - 1
#             total_loss += depth_loss_weights[stage_idx] * depth_loss
#         else:
#             total_loss += 1.0 * depth_loss

#     return total_loss, depth_loss

# 以下4个损失函数（info_entropy_loss到focal_loss_bld）都来自TransMVSNet
def info_entropy_loss(prob_volume, prob_volume_pre, mask):
    # prob_colume should be processed after SoftMax
    B,D,H,W = prob_volume.shape
    LSM = nn.LogSoftmax(dim=1)
    valid_points = torch.sum(mask, dim=[1,2])+1e-6
    entropy = -1*(torch.sum(torch.mul(prob_volume, LSM(prob_volume_pre)), dim=1)).squeeze(1)
    entropy_masked = torch.sum(torch.mul(mask, entropy), dim=[1,2])
    return torch.mean(entropy_masked / valid_points)


def entropy_loss(prob_volume, depth_gt, mask, depth_value, return_prob_map=False):
    # from AA
    mask_true = mask
    valid_pixel_num = torch.sum(mask_true, dim=[1,2]) + 1e-6
    shape = depth_gt.shape          # B,H,W

    depth_num = depth_value.shape[1]
    if len(depth_value.shape) < 3:
        depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2,3,0,1)     # B,N,H,W
    else:
        depth_value_mat = depth_value

    gt_index_image = torch.argmin(torch.abs(depth_value_mat-depth_gt.unsqueeze(1)), dim=1)

    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1) # B, 1, H, W

    # gt index map -> gt one hot volume (B x 1 x H x W )
    gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2]).type(mask_true.type()).scatter_(1, gt_index_image, 1)

    # cross entropy image (B x D X H x W)
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume + 1e-6), dim=1).squeeze(1) # B, 1, H, W

    # masked cross entropy loss
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image) # valid pixel
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])

    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num) # Origin use sum : aggregate with batch
    # winner-take-all depth map
    wta_index_map = torch.argmax(prob_volume, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_value_mat, 1, wta_index_map).squeeze(1)

    if return_prob_map:
        photometric_confidence = torch.max(prob_volume, dim=1)[0] # output shape dimension B * H * W
        return masked_cross_entropy, wta_depth_map, photometric_confidence
    return masked_cross_entropy, wta_depth_map


# def trans_mvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    total_entropy =  torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        prob_volume = stage_inputs["prob_volume"]
        depth_values = stage_inputs["depth_values"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        entropy_weight = 2.0

        entro_loss, depth_entropy = entropy_loss(prob_volume, depth_gt, mask, depth_values)
        entro_loss = entro_loss * entropy_weight
        depth_loss = F.smooth_l1_loss(depth_entropy[mask], depth_gt[mask], reduction='mean')
        total_entropy += entro_loss

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * entro_loss
        else:
            total_loss += entro_loss

    return total_loss, depth_loss, total_entropy, depth_entropy


def focal_loss_bld(inputs, depth_gt_ms, mask_ms, depth_interval, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    total_entropy = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        prob_volume = stage_inputs["prob_volume"]
        depth_values = stage_inputs["depth_values"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        entropy_weight = 2.0
        entro_loss, depth_entropy = entropy_loss(prob_volume, depth_gt, mask, depth_values)
        entro_loss = entro_loss * entropy_weight
        depth_loss = F.smooth_l1_loss(depth_entropy[mask], depth_gt[mask], reduction='mean')
        total_entropy += entro_loss

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * entro_loss
        else:
            total_loss += entro_loss

    abs_err = (depth_gt_ms['stage3'] - inputs["stage3"]["depth"]).abs()
    abs_err_scaled = abs_err /(depth_interval *192./128.)
    mask = mask_ms["stage3"]
    mask = mask > 0.5
    epe = abs_err_scaled[mask].mean()
    less1 = (abs_err_scaled[mask] < 1.).to(depth_gt_ms['stage3'].dtype).mean()
    less3 = (abs_err_scaled[mask] < 3.).to(depth_gt_ms['stage3'].dtype).mean()

    return total_loss, depth_loss, epe, less1, less3

# def trans_mvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
#     depth_loss_weights = kwargs.get("dlossw", None)
#     total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
#     total_entropy =  torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

#     for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
#         prob_volume = stage_inputs["prob_volume"]
#         depth_values = stage_inputs["depth_values"]
#         depth_gt = depth_gt_ms[stage_key]
#         mask = mask_ms[stage_key]
#         mask = mask > 0.5
#         entropy_weight = 2.0

#         entro_loss, depth_entropy = entropy_loss(prob_volume, depth_gt, mask, depth_values)
#         entro_loss = entro_loss * entropy_weight
#         depth_loss = F.smooth_l1_loss(depth_entropy[mask], depth_gt[mask], reduction='mean')
#         total_entropy += entro_loss

#         if depth_loss_weights is not None:
#             stage_idx = int(stage_key.replace("stage", "")) - 1
#             total_loss += depth_loss_weights[stage_idx] * entro_loss
#         else:
#             total_loss += entro_loss

#     return total_loss, depth_loss, total_entropy, depth_entropy


# from transmvsnet(0328), 其使用的是交叉熵损失，自己又加上图像合成损失
# def cas_mvsnet_loss(inputs,imgs,cam_para, depth_gt_ms, mask_ms, **kwargs):
#     depth_loss_weights = kwargs.get("dlossw", None)
#     total_depth_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
#     total_entropy =  torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
#     total_cpc_loss = cross_view_loss(inputs, imgs, cam_para, depth_gt_ms, depth_loss_weights)

#     for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
#         prob_volume = stage_inputs["prob_volume"]
#         depth_values = stage_inputs["depth_values"]
#         depth_gt = depth_gt_ms[stage_key]
#         mask = mask_ms[stage_key]
#         mask = mask > 0.5
#         entropy_weight = 2.0

#         entro_loss, depth_entropy = entropy_loss(prob_volume, depth_gt, mask, depth_values)
#         entro_loss = entro_loss * entropy_weight
#         depth_loss = F.smooth_l1_loss(depth_entropy[mask], depth_gt[mask], reduction='mean')
#         total_entropy += entro_loss

#         if depth_loss_weights is not None:
#             stage_idx = int(stage_key.replace("stage", "")) - 1
#             total_depth_loss += depth_loss_weights[stage_idx] * entro_loss
#         else:
#             total_depth_loss += entro_loss
#     # focal loss + cross_view_loss
#     total_loss = total_depth_loss + total_cpc_loss * 12

#     return total_loss, total_depth_loss,total_cpc_loss


def get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth=192.0, min_depth=0.0):
    #shape, (B, H, W)
    #cur_depth: (B, H, W)
    #return depth_range_values: (B, D, H, W)
    cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel)  # (B, H, W)
    cur_depth_max = (cur_depth + ndepth / 2 * depth_inteval_pixel)
    # cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel).clamp(min=0.0)   #(B, H, W)
    # cur_depth_max = (cur_depth_min + (ndepth - 1) * depth_inteval_pixel).clamp(max=max_depth)

    assert cur_depth.shape == torch.Size(shape), "cur_depth:{}, input shape:{}".format(cur_depth.shape, shape)
    new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, H, W)
    #                       (B,1,H,W)                     (1,D,1,1) * (B,1,H,W) = (B,D,H,W)
    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=cur_depth.device,
                                                                  dtype=cur_depth.dtype,
                                                                  requires_grad=False).reshape(1, -1, 1,
                                                                                               1) * new_interval.unsqueeze(1))

    return depth_range_samples


def get_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, device, dtype, shape,
                           max_depth=192.0, min_depth=0.0):
    #shape: (B, H, W)
    #cur_depth: (B, H, W) or (B, D)
    #return depth_range_samples: (B, D, H, W)
    if cur_depth.dim() == 2:
        cur_depth_min = cur_depth[:, 0]  # (B,)
        cur_depth_max = cur_depth[:, -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, )
        #                           (B,1)                                     （1，D）* （B,1） = (B,D)
        depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=device, dtype=dtype,
                                                                       requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(1)) #(B, D)
        #(B,D,1,1)->(B, D, H, W)
        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[1], shape[2]) #(B, D, H, W)

    else:

        depth_range_samples = get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth, min_depth)

    return depth_range_samples

# init_code
# def uncertainty_aware_samples(cur_depth, exp_var, ndepth, dtype, device, shape):
#     #shape, (B, H, W)
#     #cur_depth: (B, D) or (B, 1，H, W) 
#     #return depth_range_values: (B, D, H, W)
#     # torch.Size([8, 192])  torch.Size([8, 1, 512, 640])
#     print("the first stage of depth_sample")
#     print("the module of module, the shape of cur_depth is: ", cur_depth.shape)
#     if cur_depth.dim() == 2:
#         #must be the first stage
#         cur_depth_min = cur_depth[:, 0]  # (B,)
#         cur_depth_max = cur_depth[:, -1]
#         new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, )
#         depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=device, dtype=dtype,
#                                                                        requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(1)) # (B, D)
#         depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[1], shape[2]) # (B, D, H, W)
#         print("********************")
#         # torch.Size([8, 48, 512, 640])
#         print("the first stage of depth_range_samples size is: ", depth_range_samples.shape)  
#     else:
#         low_bound = -torch.min(cur_depth, exp_var)  # exp_var > 0
#         high_bound = exp_var
        
#         print("the two and thrid stage of depth_sample")
#         print("--------this is module uncertainty_aware_samples--------")
#         print("the shape of curdepth is: ", cur_depth.shape)  # torch.Size([8, 1, 512, 640])
#         print("the shape of high_bound is: ", high_bound.shape)  # torch.Size([8, 1, 512, 640])

#         # assert exp_var.min() >= 0, exp_var.min()
#         assert ndepth > 1

#         step = (high_bound - low_bound) / (float(ndepth) - 1)   # 
#         # torch.Size([8, 1, 512, 640])
#         print("the shape of step is: ", step.shape)
#         new_samps = []
#         for i in range(int(ndepth)):
#             new_samps.append(cur_depth + low_bound + step * i + eps)
#         # print("the length of new_samps is: ", new_samps.len)
#         depth_range_samples = torch.cat(new_samps, 1)
#         print("********************")
#         # torch.Size([8, 32, 512, 640])   torch.Size([8, 8, 512, 640])
#         print("the two and third stage of depth_range_samples size is: ", depth_range_samples.shape)
        
#         # assert depth_range_samples.min() >= 0, depth_range_samples.min()
#     return depth_range_samples

# adaptive interval
def uncertainty_aware_samples(cur_depth, exp_var, ndepth, dtype, device, shape):
    #shape, (B, H, W)
    #cur_depth: (B, D) or (B, 1, H, W) 
    #return depth_range_values: (B, D, H, W)
    if cur_depth.dim() == 2:
        #must be the first stage
        cur_depth_min = cur_depth[:, 0]  # (B,)
        cur_depth_max = cur_depth[:, -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, )
        depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=device, dtype=dtype,
                                                                       requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(1)) # (B, D)
        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[1], shape[2]) # (B, D, H, W)
        # print("the first stage of depth shape is: ", depth_range_samples.shape)
    else:
        low_bound = -torch.min(cur_depth, exp_var)  # exp_var > 0
        high_bound = exp_var  #(B,1,H,W)

        # assert exp_var.min() >= 0, exp_var.min()
        assert ndepth > 1

        step = (high_bound - low_bound) / (float(ndepth) - 1)   # (B,1,H,W)
        new_samps = []
        d_offset = []
        # print("exp_var:{}".format(exp_var))
        for i in range(int(ndepth)):
            d_offset.append(3*(low_bound + step*i)/(exp_var + eps))  # D个(B,1,H,W)
            new_samps.append(cur_depth + low_bound + step * i + eps)  # 先根据等间隔计算出一个假设深度
        # depth_offset = torch.cat(d_offset,1)  # (B,D,H,W)
        # 先利用Z分布求得各假设深度到上一阶段估计的深度之间的距离(权重)，然后再利用softmax进行归一化
        offset = F.softmax(torch.cat(d_offset,1), dim=1)  # (B,D,H,W)
        # print("the two and thrid stage is :")
        # print("the shape of offset is: ", offset.shape)
        # for i in range(int(ndepth)):
        #     new_samps.append(cur_depth + low_bound + step * i )
        # depth_range_samples = torch.cat(new_samps, 1)
        #                      (B,D,H,W) + (B,D,H,W)*(B,1,H,W)
        depth_range_samples = torch.cat(new_samps, 1) + offset * step  # 在原等间隔假设深度的基础上+自适应间隔
        # print("the shape of offset is: ", depth_range_samples.shape)
        # assert depth_range_samples.min() >= 0, depth_range_samples.min()
    return depth_range_samples

if __name__ == "__main__":
    # some testing code, just IGNORE it
    import sys
    sys.path.append("../")
    from datasets import find_dataset_def
    from torch.utils.data import DataLoader
    import numpy as np
    import cv2
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    # MVSDataset = find_dataset_def("colmap")
    # dataset = MVSDataset("../data/results/ford/num10_1/", 3, 'test',
    #                      128, interval_scale=1.06, max_h=1250, max_w=1024)

    MVSDataset = find_dataset_def("dtu_yao")
    num_depth = 48
    dataset = MVSDataset("../data/DTU/mvs_training/dtu/", '../lists/dtu/train.txt', 'train',
                         3, num_depth, interval_scale=1.06 * 192 / num_depth)

    dataloader = DataLoader(dataset, batch_size=1)
    item = next(iter(dataloader))

    imgs = item["imgs"][:, :, :, ::4, ::4]  #(B, N, 3, H, W)
    # imgs = item["imgs"][:, :, :, :, :]
    proj_matrices = item["proj_matrices"]   #(B, N, 2, 4, 4) dim=N: N view; dim=2: index 0 for extr, 1 for intric
    proj_matrices[:, :, 1, :2, :] = proj_matrices[:, :, 1, :2, :]
    # proj_matrices[:, :, 1, :2, :] = proj_matrices[:, :, 1, :2, :] * 4
    depth_values = item["depth_values"]     #(B, D)

    imgs = torch.unbind(imgs, 1)
    proj_matrices = torch.unbind(proj_matrices, 1)
    ref_img, src_imgs = imgs[0], imgs[1:]
    ref_proj, src_proj = proj_matrices[0], proj_matrices[1:][0]  #only vis first view

    src_proj_new = src_proj[:, 0].clone()
    src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
    ref_proj_new = ref_proj[:, 0].clone()
    ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])

    warped_imgs = homo_warping(src_imgs[0], src_proj_new, ref_proj_new, depth_values)

    ref_img_np = ref_img.permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255
    cv2.imwrite('../tmp/ref.png', ref_img_np)
    cv2.imwrite('../tmp/src.png', src_imgs[0].permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255)

    for i in range(warped_imgs.shape[2]):
        warped_img = warped_imgs[:, :, i, :, :].permute([0, 2, 3, 1]).contiguous()
        img_np = warped_img[0].detach().cpu().numpy()
        img_np = img_np[:, :, ::-1] * 255

        alpha = 0.5
        beta = 1 - alpha
        gamma = 0
        img_add = cv2.addWeighted(ref_img_np, alpha, img_np, beta, gamma)
        cv2.imwrite('../tmp/tmp{}.png'.format(i), np.hstack([ref_img_np, img_np, img_add])) #* ratio + img_np*(1-ratio)]))