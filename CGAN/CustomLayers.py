# -*- coding: utf-8 -*-
# @Time    : 2021/4/30 0030 23:03
# @Author  : Daniel Zhang
# @Email   : zhangdan_nuaa@163.com
# @File    : CustomLayers.py
# @Software: PyCharm


import torch
import torch.nn as nn

from utils import weight_init


class Expression(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class Residual_Block(nn.Module):
    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(Residual_Block, self).__init__()

        midc = int(outc * scale)

        if inc is not outc:
            self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0,
                                         groups=1, bias=False)
        else:
            self.conv_expand = None

        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.weight_init()

    def weight_init(self):
        initializer = weight_init
        for block in self._modules:
            initializer(self._modules[block])

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output = self.relu2(torch.add(output, identity_data))
        return output


class ToRGB(nn.Module):
    '''feature map to RGB'''
    def __init__(self, in_channels, cdim, feature_size):
        super().__init__()

        cc, sz = in_channels, feature_size
        self.main = nn.Sequential()
        self.main.add_module('toRGB_in_{}'.format(sz), nn.Conv2d(cc, cdim, 1, 1, 0))

        self.weight_init()

    def weight_init(self):
        initializer = weight_init
        for block in self._modules['main']:
            initializer(block)

    def forward(self, x):
        y = self.main(x)
        return y


class GenInitialBlock(nn.Module):
    def __init__(self, cdim=3, hdim=512, class_num=10, is_mnist=False):
        super().__init__()
        self.is_mnist = is_mnist

        self.init = nn.Sequential()
        self.init.add_module('init_linear', nn.Linear(in_features=hdim + class_num, out_features=hdim * 4 * 4))
        self.init.add_module('init_reshape', Expression(lambda x: x.reshape(-1, hdim, 4, 4)))
        self.init.add_module('init_bn', nn.BatchNorm2d(hdim))
        self.init.add_module('init_lr', nn.LeakyReLU(0.2, True))

        self.mnist_deconv = nn.Sequential()
        self.mnist_deconv.add_module('mnist_deconv', nn.ConvTranspose2d(in_channels=hdim,
                                                                        out_channels=hdim,
                                                                        kernel_size=5,
                                                                        stride=2,
                                                                        padding=2,
                                                                        output_padding=0))
        # toRGB
        self.toRGB = ToRGB(hdim, cdim, 7 if is_mnist else 4)

        self.weight_init()

    def weight_init(self):
        initializer = weight_init
        for block in self._modules['init']:
            initializer(block)
        for block in self._modules['mnist_deconv']:
            initializer(block)

    def forward(self, z, label):
        y = torch.cat((z.view(z.size(0), -1), label), dim=1)
        y = self.init(y)
        if self.is_mnist:
            y = self.mnist_deconv(y)
        rgb = self.toRGB(y)
        return y, rgb


class GenGeneralResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, feature_size, cdim, class_num=10):
        super().__init__()
        cc, ch, sz = in_channels, out_channels, feature_size

        self.main = nn.Sequential()
        self.main.add_module('up_to_{}'.format(sz * 2),
                             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.main.add_module('res_in_{}'.format(sz), Residual_Block(cc + class_num, ch, scale=1.0))

        # toRGB
        self.toRGB = ToRGB(ch, cdim, sz)

        self.weight_init()

    def weight_init(self):
        initializer = weight_init
        for block in self._modules['main']:
            initializer(block)

    def forward(self, x, label):
        label = label.reshape((label.shape[0], label.shape[1], 1, 1)).repeat([1, 1, x.shape[2], x.shape[3]])
        y = torch.cat((x, label), dim=1)
        y = self.main(y)
        rgb = self.toRGB(y)
        return y, rgb


class GenFinalResBlock(nn.Module):
    def __init__(self, in_channels, feature_size, cdim, class_num=10):
        super().__init__()
        cc, sz = in_channels, feature_size

        self.main = nn.Sequential()
        self.main.add_module('res_in_{}'.format(sz), Residual_Block(cc + class_num, cc, scale=1.0))
        self.main.add_module('predict', nn.Conv2d(cc, cdim, 5, 1, 2))

        self.weight_init()

    def weight_init(self):
        initializer = weight_init
        for block in self._modules['main']:
            initializer(block)

    def forward(self, x, label):
        label = label.reshape((label.shape[0], label.shape[1], 1, 1)).repeat([1, 1, x.shape[2], x.shape[3]])
        y = torch.cat((x, label), dim=1)
        y = self.main(y)
        return y, None


class DisInitialBlock(nn.Module):
    def __init__(self, cdim, out_channels, feature_size, class_num=10):
        super().__init__()
        cc, sz = out_channels, feature_size

        self.main = nn.Sequential()
        self.main.add_module('conv_in_{}'.format(sz), nn.Conv2d(cdim + class_num, cc, 5, 1, 2, bias=False))
        self.main.add_module('bn_in_{}'.format(sz), nn.BatchNorm2d(cc))
        self.main.add_module('lr_in_{}'.format(sz), nn.LeakyReLU(0.2, True))
        self.main.add_module('res_in_{}'.format(sz), Residual_Block(cc, cc, scale=1.0))

        self.weight_init()

    def weight_init(self):
        initializer = weight_init
        for block in self._modules['main']:
            initializer(block)

    def forward(self, x, rgb, label):
        label = label.reshape((label.shape[0], label.shape[1], 1, 1)).repeat([1, 1, x.shape[2], x.shape[3]])
        y = torch.cat((x, label), dim=1)
        y = self.main(y)
        return y


class DisGeneralResBlock(nn.Module):
    def __init__(self, cdim, in_channels, out_channels, feature_size, class_num=10):
        super().__init__()
        cc, ch, sz = in_channels, out_channels, feature_size

        self.main = nn.Sequential()
        self.main.add_module('res_in_{}'.format(sz), Residual_Block(cc + class_num, ch, scale=1.0))
        self.main.add_module('down_to_{}'.format(sz // 2), nn.AvgPool2d(2))

        # FromRGB
        self.fromRGB = FromRGB(cdim, ch, sz // 2)

        self.weight_init()

    def weight_init(self):
        initializer = weight_init
        for block in self._modules['main']:
            initializer(block)

    def forward(self, x, rgb, label):
        label = label.reshape((label.shape[0], label.shape[1], 1, 1)).repeat([1, 1, x.shape[2], x.shape[3]])
        y = torch.cat((x, label), dim=1)
        y = self.main(y)
        return y + self.fromRGB(rgb)


class DisFinalBlock(nn.Module):
    def __init__(self, cdim, in_channels, out_channels,
                 feature_size, class_num=10, is_mnist=False):
        super().__init__()
        cc, ch, sz = in_channels, out_channels, feature_size

        self.main = nn.Sequential()
        self.main.add_module('res_in_{}'.format(sz), Residual_Block(cc + class_num, ch, scale=1.0))
        self.main.add_module('down_to_{}'.format(sz // 2), nn.AvgPool2d(2))
        self.to_logit = nn.Linear((ch + 1) * 7 * 7, 1) if is_mnist \
            else nn.Linear((ch + 1) * 4 * 4, 1)

        self.batch_discriminator = MinibatchStdDev()

        # FromRGB
        self.fromRGB = FromRGB(cdim, ch, sz // 2)

        self.weight_init()

    def weight_init(self):
        initializer = weight_init
        for block in self._modules['main']:
            initializer(block)
        initializer(self._modules['to_logit'])

    def forward(self, x, rgb, label):
        label = label.reshape((label.shape[0], label.shape[1], 1, 1)).repeat([1, 1, x.shape[2], x.shape[3]])
        y = torch.cat((x, label), dim=1)
        y = self.main(y) + self.fromRGB(rgb)

        y = self.batch_discriminator(y)
        y = y.view(y.size(0), -1)
        return self.to_logit(y)


class FromRGB(nn.Module):
    def __init__(self, cdim, out_channels, feature_size):
        super().__init__()
        ch, sz = out_channels, feature_size

        self.main = nn.Sequential()
        self.main.add_module('FromRGB_conv_in_{}'.format(sz), nn.Conv2d(cdim, ch, 1, 1, 0))

        self.weight_init()

    def weight_init(self):
        initializer = weight_init
        for block in self._modules['main']:
            initializer(block)

    def forward(self, rgb):
        y = self.main(rgb)
        return y


class MinibatchStdDev(nn.Module):
    """
    Minibatch standard deviation layer for the discriminator
    """
    def __init__(self):
        """
        derived class constructor
        """
        super().__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape

        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)

        # [1 x C x H x W]  Calc standard deviation over batch
        y = torch.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size, 1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)

        # return the computed values:
        return y
