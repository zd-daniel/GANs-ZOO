# -*- coding: utf-8 -*-
# @Time    : 2021/4/30 0030 22:33
# @Author  : Daniel Zhang
# @Email   : zhangdan_nuaa@163.com
# @File    : main.py
# @Software: PyCharm


import argparse
import os
import numpy as np
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from datasets import load_data
from utils import EMA, load_model, save_checkpoint

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_arg():
    parser = argparse.ArgumentParser('参数管理')
    parser.add_argument('--dataroot', default="", type=str, help='data path')
    parser.add_argument('--outfile', default='results', type=str, help='output path')
    parser.add_argument("--pretrained", default="", type=str, help="path for pre-training model")

    parser.add_argument('--noise_dim', type=int, default=128, help='latent space')
    parser.add_argument('--channels', default="32, 64", type=str,
                        help='channels for each layer')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--g_lr', type=float, default=0.001, help='learning rate for generator')
    parser.add_argument('--d_lr', type=float, default=0.001, help='learning rate for discriminator')
    parser.add_argument('--lr_decay', type=float, default=1., help='decay')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2')

    parser.add_argument('--input_height', type=int, default=28, help='input image size')
    parser.add_argument('--input_width', type=int, default=None, help='input image size')
    parser.add_argument('--output_height', type=int, default=28, help='output image size')
    parser.add_argument('--output_width', type=int, default=None, help='output image size')
    parser.add_argument('--crop_height', type=int, default=None, help='crop image size')
    parser.add_argument('--crop_width', type=int, default=None, help='crop image size')

    parser.add_argument('--weight_ema', type=float, default=0.995, help='滑动平均系数')
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument('--nEpochs', type=int, default=600, help='epochs')

    return parser.parse_known_args()[0]


def main(datatype='mnist', mode='ACGAN'):
    assert datatype in ['mnist', 'cifar10', 'celebAHQ'], '未知数据集'
    assert mode in ['ACGAN', 'CGAN'], '未知GAN'

    config = parse_arg()
    disp_str = ''
    for attr in sorted(dir(config), key=lambda x: len(x)):
        if not attr.startswith('_'):
            disp_str += ' {} : {}\n'.format(attr, getattr(config, attr))
    print(disp_str)

    try:
        config.outfile = mode + '/' + config.outfile
        os.makedirs(config.outfile)
        print('mkdir:', config.outfile)
    except OSError:
        pass

    seed = np.random.randint(0, 10000)
    print("Random Seed: ", seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True

    # -----------------load dataset--------------------------
    if datatype == 'mnist':
        config.dataroot = 'G:/Dataset/mnist/'
    elif datatype == 'cifar10':
        config.dataroot = 'G:/Dataset/cifar10/'
    elif datatype == 'celebAHQ':
        config.dataroot = 'G:/Dataset/celebAHQ/celeba-64'
    trainset, trainloader, testset, testloader, classes = \
        load_data(config.dataroot, datatype=datatype, batch_size=config.batch_size)

    # --------------build models -------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    str_to_list = lambda x: [int(xi) for xi in x.split(',')]
    if mode == 'ACGAN':
        from ACGAN.networks import ACGAN
        from ACGAN.train import model_train, model_eval
        model = ACGAN(cdim=3, hdim=config.noise_dim,
                      channels=str_to_list(config.channels), image_size=config.output_height,
                      class_num=10, is_mnist=False).to(device)
    elif mode == 'CGAN':
        from CGAN.networks import CGAN
        from CGAN.train import model_train, model_eval
        model = CGAN(cdim=1, hdim=config.noise_dim,
                     channels=str_to_list(config.channels), image_size=config.output_height,
                     class_num=10, is_mnist=True).to(device)
    ema = EMA(model, config.weight_ema)
    ema.register()
    if config.pretrained:
        load_model(model, ema, mode + '/' + config.pretrained)
    # print(model)

    d_optimizer = torch.optim.Adam(model.dis.parameters(), config.d_lr, betas=(config.beta1, config.beta2))
    g_optimizer = torch.optim.Adam(model.gen.parameters(), config.g_lr, betas=(config.beta1, config.beta2))

    fix_noise = torch.zeros(100, config.noise_dim).uniform_(-1, 1).to(device)
    fix_label = torch.arange(0, 10).reshape(-1, 1).repeat([1, 10]).flatten().to(device)
    start_time = time.time()
    cur_iter = 0
    for epoch in range(config.start_epoch, config.nEpochs):
        model.train()
        for iteration, (real_images, real_labels) in enumerate(trainloader):
            real_images = real_images.to(device).requires_grad_(True)
            real_labels = real_labels.to(device)

            info = "\n====> Cur_iter: [{}]: Epoch[{}]({}/{}): time: {:4.4f}: ".format(cur_iter, epoch, iteration,
                                                                                      len(trainloader),
                                                                                      time.time() - start_time)

            model_train(model, real_images, real_labels, d_optimizer, g_optimizer, config, info, device)
            ema.update()
            cur_iter += 1

        save_checkpoint(model, ema, epoch, mode)
        ema.apply_shadow()
        model_eval(model, fix_noise, fix_label, config, epoch)
        ema.restore()


if __name__ == "__main__":
    main(datatype='mnist', mode='CGAN')
