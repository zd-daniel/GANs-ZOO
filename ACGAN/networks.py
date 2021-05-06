# -*- coding: utf-8 -*-
# @Time    : 2021/4/30 0030 23:04
# @Author  : Daniel Zhang
# @Email   : zhangdan_nuaa@163.com
# @File    : networks.py
# @Software: PyCharm


import torch
import torch.nn as nn
from torch.nn.parallel import data_parallel

from ACGAN.CustomLayers import GenInitialBlock, GenGeneralResBlock, GenFinalResBlock, \
    DisInitialBlock, DisGeneralResBlock, DisFinalBlock


class Generator(nn.Module):
    def __init__(self, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512],
                 image_size=256, class_num=10, is_mnist=False):
        super().__init__()
        self.class_num = class_num
        if is_mnist:
            assert (2 ** len(channels)) * 7 == image_size, '网络层数与图像大小不匹配'
        else:
            assert (2 ** len(channels)) * 4 == image_size, '网络层数与图像大小不匹配'

        self.layers = nn.ModuleList()
        self.layers.append(GenInitialBlock(cdim, hdim, class_num, is_mnist))

        cc, sz = hdim, 4
        for ch in channels[::-1]:
            self.layers.append(GenGeneralResBlock(in_channels=cc, out_channels=ch, feature_size=sz,
                                                  cdim=cdim, class_num=class_num))
            cc, sz = ch, sz * 2
        self.layers.append(GenFinalResBlock(in_channels=cc, feature_size=sz, cdim=cdim, class_num=class_num))

        # 残差上采样
        self.upsampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, z, label):
        label_onehot = torch.zeros(len(label), self.class_num).to(label.device).scatter_(1, label.reshape(-1, 1), 1)
        y, out = z, None
        for block in self.layers:
            y, rgb = block(y, label_onehot)

            if rgb is None:
                out += y
                break
            if out is None:
                out = rgb
            else:
                out = self.upsampler(out) + rgb
        return out


class Discriminator(nn.Module):
    def __init__(self, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512],
                 image_size=256, class_num=10, is_mnist=False):
        super().__init__()
        if is_mnist:
            assert (2 ** len(channels)) * 7 == image_size, '网络层数与图像大小不匹配'
        else:
            assert (2 ** len(channels)) * 4 == image_size, '网络层数与图像大小不匹配'

        self.layers = nn.ModuleList()
        cc, sz = channels[0], image_size
        self.layers.append(DisInitialBlock(cdim, out_channels=cc, feature_size=sz))
        for ch in channels[1:]:
            self.layers.append(DisGeneralResBlock(cdim, cc, ch, sz // 2))
            cc, sz = ch, sz // 2
        self.layers.append(DisFinalBlock(cdim, in_channels=cc, out_channels=hdim,
                                         feature_size=sz // 2, class_num=class_num, is_mnist=is_mnist))

        # 残差下采样
        self.downsampler = nn.AvgPool2d(2)

    def forward(self, rgb):
        y = rgb
        for i, block in enumerate(self.layers):
            if i >= 1:
                rgb = self.downsampler(rgb)
            y = block(y, rgb)
        return y


class ACGAN(nn.Module):
    def __init__(self, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512],
                 image_size=256, class_num=10, is_mnist=False):
        super().__init__()

        self.hdim = hdim
        self.dis = Discriminator(cdim, hdim, channels, image_size, class_num, is_mnist)
        self.gen = Generator(cdim, hdim, channels, image_size, class_num, is_mnist)

    def discriminate(self, x):
        out = data_parallel(self.dis, x)
        logits, probs = out[:, 0], out[:, 1:]
        return logits, probs

    def generate(self, z, label):
        y = data_parallel(self.gen, (z, label))
        return y


if __name__ == '__main__':
    latent_code = torch.zeros((2, 128)).uniform_(-1, 1)
    class_num = 10
    label = torch.randint(0, 10, size=(2, ))

    gen = Generator(cdim=3, hdim=128, channels=[32, 64, 128],
                    image_size=32, class_num=class_num,  is_mnist=False)
    img = gen(latent_code, label)
    print(img.shape)

    # n = 0
    # for name, param in gen.named_parameters():
    #     # print(name)
    #     # print(param.shape)
    #     print(torch.mean(param))
    #     print(torch.std(param))
    #     print('-----')
    #     n += torch.prod(torch.tensor(param.shape))
    # print(n)

    dis = Discriminator(cdim=3, hdim=128, channels=[32, 64, 128],
                        image_size=32, class_num=class_num, is_mnist=False)
    logit = dis(img)
    print(logit.shape)

    # n = 0
    # for name, param in dis.named_parameters():
    #     # print(name)
    #     # print(param.shape)
    #     print(torch.mean(param))
    #     print(torch.std(param))
    #     print('-----')
    #     n += torch.prod(torch.tensor(param.shape))
    # print(n)
