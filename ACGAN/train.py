# -*- coding: utf-8 -*-
# @Time    : 2021/5/5 0005 14:18
# @Author  : Daniel Zhang
# @Email   : zhangdan_nuaa@163.com
# @File    : train.py
# @Software: PyCharm


import torch
import torch.nn as nn
from torchvision.utils import save_image


logit_criterion = nn.BCEWithLogitsLoss(reduction='mean')
prob_criterion = nn.CrossEntropyLoss(reduction='mean')


def model_train(model, real_images, real_labels, d_optimizer, g_optimizer, config, info, device):
    noise = torch.zeros((config.batch_size, config.noise_dim)).uniform_(-1, 1).to(device)
    fake_images = model.generate(noise, real_labels)

    real_logits, real_probs = model.discriminate(real_images)
    fake_logits, fake_probs = model.discriminate(fake_images)

    d_logits_loss = logit_criterion(real_logits, torch.ones_like(real_logits) - 0.001) + \
                    logit_criterion(fake_logits, torch.zeros_like(fake_logits) + 0.001)
    probs_loss = prob_criterion(real_probs, real_labels) + \
                 prob_criterion(fake_probs, real_labels)
    d_optimizer.zero_grad()
    d_loss = d_logits_loss + probs_loss
    d_loss.backward(retain_graph=True)
    d_optimizer.step()

    g_logits_loss = logit_criterion(fake_logits, torch.ones_like(real_logits) - 0.001)
    g_optimizer.zero_grad()
    g_loss = g_logits_loss + probs_loss
    g_loss.backward()
    g_optimizer.step()

    info += 'd_loss: {:.4f}, ' \
            'g_loss: {:.4f}, ' \
            'probs_loss: {:.4f}, ' \
            'd_logits: {:.4f}, ' \
            'g_logits: {:.4f}'.format(d_loss.item(),
                                      g_loss.item(),
                                      probs_loss.item(),
                                      d_logits_loss.item(),
                                      g_logits_loss.item())
    print(info)


def model_eval(model, fix_noise, fix_label, config, epoch):
    rescaling_inv = lambda x: .5 * x + .5

    model.eval()
    fix_images = model.generate(fix_noise, fix_label)
    fix_images = rescaling_inv(fix_images)
    save_image(fix_images.cpu().data, config.outfile + '/{}.png'.format(epoch), nrow=10)
