import torch
import torch.nn as nn
import torch.nn.functional as F



class PartialConv2d(nn.Conv2d):

    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        self.return_mask = True

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask=None):

        if mask is not None or self.last_size != (input.data.shape[2], input.data.shape[3]):
            self.last_size = (input.data.shape[2], input.data.shape[3])

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        if self.update_mask.type() != input.type() or self.mask_ratio.type() != input.type():
            self.update_mask.to(input)
            self.mask_ratio.to(input)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


# --------------------------
# PConv-BatchNorm-Activation
# --------------------------
class PConvBNActiv(nn.Module):

    def __init__(self, in_channels, out_channels, bn=True, sample='none-3', activ='relu', bias=False):
        super(PConvBNActiv, self).__init__()

        if sample == 'down-7':
            self.increase_channels = nn.Conv2d(in_channels, out_channels, 1)

            self.large_res = self.res_block(in_channels, out_channels, is_large_small='large', kernel_size=31, stride=1,
                                            padding=15, groups=in_channels)

            self.small_res = self.res_block(in_channels, out_channels, is_large_small='small')

            self.conv = PartialConv2d(out_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=bias,
                                      multi_channel=True)

            self.se_res = SELayer(out_channels, 16)
            self.se_dconv = SELayer(out_channels, 16)
        elif sample == 'down-5':
            self.increase_channels = nn.Conv2d(in_channels, out_channels, 1)
            self.large_res = self.res_block(in_channels, out_channels, is_large_small='large', kernel_size=13, stride=1,
                                            padding=6, groups=in_channels)

            self.small_res = self.res_block(in_channels, out_channels, is_large_small='small')

            self.conv = PartialConv2d(out_channels, out_channels, kernel_size=5, stride=2, padding=2, bias=bias,
                                     multi_channel=True)
            self.se_res = SELayer(out_channels, 16)
            self.se_dconv = SELayer(out_channels, 16)
        elif sample == 'down-3':
            self.increase_channels = nn.Conv2d(in_channels, out_channels, 1)
            self.large_res = self.res_block(in_channels, out_channels, is_large_small='large', kernel_size=13, stride=1,
                                            padding=6, groups=in_channels)

            self.small_res = self.res_block(in_channels, out_channels, is_large_small='small')

            self.conv = PartialConv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=bias,
                                      multi_channel=True)
            self.se_res = SELayer(out_channels, 16)
            self.se_dconv = SELayer(out_channels, 16)
        else:
            self.large_res = None
            self.small_res = None
            self.decrease_channels = None

            self.conv = PartialConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias,
                                      multi_channel=True)

        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def res_block(self, in_channels, out_channels, is_large_small=None, kernel_size=None, stride=None, padding=None,
                  groups=None):
        if is_large_small == 'large':
            return Depthwise_separable_conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=padding, groups=groups)
        if is_large_small == 'small':
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, images, masks):

        if self.small_res is not None:  # 判断上采样的条件

            images_l, masks_l = self.large_res(images, masks)  # 31,1,15

            images_s = self.small_res(images)  # 3,1,1
            masks_s = self.small_res(masks)  # 3,1,1

            images_l = self.se_res(images_l)
            masks_l = self.se_res(masks_l)

            images_s = self.se_res(images_s)
            masks_s = self.se_res(masks_s)

            images_i = self.increase_channels(images)
            masks_i = self.increase_channels(masks)

            images = images_i + images_l + images_s  #
            masks = masks_i + masks_l + masks_s  #

        images, masks = self.conv(images, masks)
        if hasattr(self, 'bn'):
            images = self.bn(images)
        if hasattr(self, 'activation'):
            images = self.activation(images)

        return images, masks


class Depthwise_separable_conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups):
        super(Depthwise_separable_conv, self).__init__()

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
            ),
            #nn.SyncBatchNorm(out_channels),
            nn.GroupNorm(num_channels=out_channels,num_groups=4),
            nn.ReLU6(),
        )
        # def __init__(self,in_channels,out_channels):
        # super(Depthwise_separable_conv)
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
            ),
            #nn.SyncBatchNorm(out_channels),
            nn.GroupNorm(num_channels=out_channels,num_groups=4),
            nn.ReLU6(),
        )
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, images, masks):
        images1 = self.depthwise_conv(images)
        masks1 = self.depthwise_conv(masks)

        images2 = self.pointwise_conv(images1)
        masks2 = self.pointwise_conv(masks1)

        return images2, masks2


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            # 第一次全连接，降低维度
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            # 第二次全连接，恢复维度
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 对应Squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # 对应Excitation操作
        return x * y.expand_as(x)  # 把权重矩阵赋予到特征图
