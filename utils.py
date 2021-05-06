# -*- coding: utf-8 -*-
# @Time    : 2021/4/30 0030 22:37
# @Author  : Daniel Zhang
# @Email   : zhangdan_nuaa@163.com
# @File    : utils.py
# @Software: PyCharm


import os
import torch
import torch.nn as nn


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg", ".bmp"])


def load_model(model, ema, pretrained):
    weights = torch.load(pretrained)

    pretrained_dict = weights['model']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    if 'ema_shadow' in weights.keys():
        ema.shadow = weights['ema_shadow']


def save_checkpoint(model, ema, epoch, mode):
    state = {'model': model.state_dict(), 'ema_shadow': ema.shadow}
    model_out_path = mode + "/model/" + "model_epoch_{}.pth".format(epoch)
    if not os.path.exists(mode + "/model/"):
        os.makedirs(mode + "/model/")

    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


class EMA:
    '''ExponentialMovingAverage'''
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data.copy_(self.backup[name])
        self.backup = {}


def weight_init(m, mode='kaiming'):
    assert mode in ['normal', 'xavier', 'kaiming'], '未知初始化方法'

    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        if mode == 'normal':
            nn.init.normal_(m.weight, mean=0, std=0.002)
        elif mode == 'xavier':
            nn.init.xavier_normal_(m.weight)
        else:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
