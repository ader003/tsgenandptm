# -*- coding: utf-8 -*-
"""
@author:  <blinded>
"""


import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from scipy import signal
from .util import _get_checkpoint
from .util import _get_start_epoch
from ..base_datafeeder import BaseDatafeeder


class AfterPoolLayer(nn.Module):
    def __init__(self):
        super(AfterPoolLayer, self).__init__()

    def forward(self, x):
        return x[:, :, 0]


class BeforeUpsampleLayer(nn.Module):
    def __init__(self):
        super(BeforeUpsampleLayer, self).__init__()

    def forward(self, x):
        return torch.unsqueeze(x, 2)


class Block(nn.Module):
    def __init__(self, in_dim, out_dim, block_type='down'):
        super(Block, self).__init__()
        if block_type == 'up':
            self.conv_00 = nn.Conv1d(
                in_dim, out_dim, 3, padding=1)
            self.transform_00 = nn.ConvTranspose1d(
                out_dim, out_dim, 4, stride=2, padding=1)
        elif block_type == 'down':
            self.conv_00 = nn.Conv1d(
                in_dim, out_dim, 3, padding=1)
            self.transform_00 = nn.Conv1d(
                out_dim, out_dim, 4, stride=2, padding=1)
        self.conv_01 = nn.Conv1d(
            out_dim, out_dim, 3, padding=1)
        self.norm_00 = nn.BatchNorm1d(out_dim)
        self.norm_01 = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        h = self.conv_00(x)
        h = nn.ReLU()(h)
        h = self.norm_00(h)

        h = self.conv_01(h)
        h = nn.ReLU()(h)
        h = self.norm_01(h)

        h = self.transform_00(h)
        return h


class GANNet(nn.Module):
    def __init__(self, in_dim, in_len, n_dim=64, n_layer=5):
        super(GANNet, self).__init__()
        self.in_dim = in_dim
        self.n_layer = n_layer

        h_len = in_len // (2 ** self.n_layer)
        self.h_len = h_len

        discriminator = OrderedDict()
        encoder = OrderedDict()

        in_dim_ = in_dim
        out_dim = n_dim
        discriminator['conv_00'] = nn.Conv1d(
            in_dim_, out_dim, 3, padding=1)
        encoder['conv_00'] = nn.Conv1d(
            in_dim_, out_dim, 3, padding=1)
        for i in range(n_layer):
            in_dim_ = out_dim
            out_dim = out_dim * 2
            discriminator[f'block_{i:02d}'] = Block(
                in_dim_, out_dim, block_type='down')
            encoder[f'block_{i:02d}'] = Block(
                in_dim_, out_dim, block_type='down')

        discriminator['pool'] = nn.AdaptiveAvgPool1d(1)
        discriminator['after_pool'] = AfterPoolLayer()
        discriminator['linear'] = nn.Linear(out_dim, 2)
        self.discriminator = nn.Sequential(discriminator)

        encoder['pool'] = nn.AdaptiveAvgPool1d(1)
        encoder['after_pool'] = AfterPoolLayer()
        encoder['linear'] = nn.Linear(out_dim, out_dim)
        self.encoder = nn.Sequential(encoder)
        self.laten_dim = out_dim

        decoder = OrderedDict()
        decoder['before_upsample'] = BeforeUpsampleLayer()
        decoder['upsample'] = nn.Upsample(size=h_len)
        for i in range(n_layer):
            in_dim_ = out_dim
            out_dim = out_dim // 2
            decoder[f'block_{i:02d}'] = Block(
                in_dim_, out_dim, block_type='up')
        decoder['conv_01'] = nn.Conv1d(
            out_dim, in_dim, 1)
        self.decoder = nn.Sequential(decoder)
        self.dummy = nn.Parameter(torch.empty(0))

    def forward(self, x):
        device = self.dummy.device
        laten_dim = self.laten_dim

        z_fake = torch.randn(
            (x.size(0), laten_dim, ), device=device)
        x_hat_fake = self.decoder(z_fake)

        y_hat_real = self.discriminator(x)
        y_hat_fake = self.discriminator(x_hat_fake)
        return y_hat_real, y_hat_fake

    def get_parameter_set(self):
        param_rec = (list(self.encoder.parameters()) +
                     list(self.decoder.parameters()))
        param_dis = self.discriminator.parameters()
        param_gen = self.decoder.parameters()
        return param_rec, param_dis, param_gen

    def get_loss(self, x, loss_type):
        device = self.dummy.device
        if loss_type == 'rec':
            z_real = self.encoder(x)
            x_hat_real = self.decoder(z_real)
            loss = F.mse_loss(x, x_hat_real)
        elif loss_type == 'dis':
            y_hat_real, y_hat_fake = self.forward(x)
            # y_real = torch.ones((x.size(0), 1)).to(device)
            # y_fake = torch.zeros((x.size(0), 1)).to(device)
            # criterion = nn.BCEWithLogitsLoss()
            # loss = (criterion(y_hat_real, y_real) +
            #         criterion(y_hat_fake, y_fake))
            loss = (-torch.mean(nn.Sigmoid()(y_hat_real)) +
                    torch.mean(nn.Sigmoid()(y_hat_fake)))
        elif loss_type == 'gen':
            y_hat_real, y_hat_fake = self.forward(x)
            # y_real = torch.ones((x.size(0), 1)).to(device)
            # y_fake = torch.zeros((x.size(0), 1)).to(device)
            # criterion = nn.BCEWithLogitsLoss()
            # loss = (criterion(y_hat_real, y_fake) +
            #         criterion(y_hat_fake, y_real))
            loss = (-torch.mean(nn.Sigmoid()(y_hat_fake)) +
                    torch.mean(nn.Sigmoid()(y_hat_real)))
        return loss

    @torch.no_grad()
    def sample(self, n_sample):
        device = self.dummy.device
        laten_dim = self.laten_dim
        z = torch.randn(
            (n_sample, laten_dim), device=device)
        x_hat = self.decoder(z)
        return x_hat


def adversarial_train(data, model_path, model_config, device):
    in_dim = data.shape[1]
    data = copy.deepcopy(data)

    cur_len = data.shape[2]
    if cur_len < 32:
        new_len = 32
    else:
        new_len = 2 ** np.ceil(np.log(cur_len) / np.log(2))
        new_len = int(new_len)
    data = signal.resample(
        data, new_len, axis=2)

    data = data - np.min(data)
    data = data / np.max(data)
    data = data * 2.0 - 1.0

    n_dim = int(model_config['n_dim'])
    n_layer = int(model_config['n_layer'])

    ckpt_gap = int(model_config['ckpt_gap'])
    n_epoch = int(model_config['n_epoch'])
    batch_size = int(model_config['batch_size'])
    lr = float(model_config['lr'])
    feeder = BaseDatafeeder(data, batch_size)
    n_batch = feeder.n_batch

    model = GANNet(in_dim, new_len, n_dim=n_dim, n_layer=n_layer)
    model.to(device)
    model.train()

    ckpts = _get_checkpoint(ckpt_gap, n_epoch)
    start_epoch = _get_start_epoch(model_path, ckpts)

    param_rec, param_dis, param_gen = model.get_parameter_set()
    optimizer_rec = torch.optim.AdamW(
        param_rec, lr=lr)
    optimizer_dis = torch.optim.AdamW(
        param_dis, lr=lr)
    optimizer_gen = torch.optim.AdamW(
        param_gen, lr=lr)

    loss_train = np.zeros(n_epoch)
    toc_train = np.zeros(n_epoch)
    for i in range(start_epoch, n_epoch):
        if start_epoch != 0 and i == start_epoch:
            print(f'resume training from epoch {i + 1:d}')
        model_path_i = model_path.format(i)
        if os.path.isfile(model_path_i):
            print(f'loading {model_path_i}')
            pkl = torch.load(model_path_i, map_location='cpu')
            loss_train = pkl['loss_train']
            toc_train = pkl['toc_train']
            loss_epoch = loss_train[i]
            toc_epoch = toc_train[i]

            model.load_state_dict(
                pkl['model_state_dict'])
            model.to(device)
            model.train()

            optimizer_rec.load_state_dict(
                pkl['optimizer_rec_state_dict'])
            optimizer_dis.load_state_dict(
                pkl['optimizer_dis_state_dict'])
            optimizer_gen.load_state_dict(
                pkl['optimizer_gen_state_dict'])

        tic = time.time()
        loss_epoch = 0
        for j in range(n_batch):
            batch = feeder.get_batch()
            batch_torch = torch.from_numpy(batch)
            batch_torch = batch_torch.to(device, dtype=torch.float)

            optimizer_rec.zero_grad()
            loss = model.get_loss(batch_torch, 'rec')
            loss.backward()
            optimizer_rec.step()
            loss_epoch += loss.item()

            optimizer_dis.zero_grad()
            loss = model.get_loss(batch_torch, 'dis')
            loss.backward()
            optimizer_dis.step()
            loss_epoch += loss.item()

            optimizer_gen.zero_grad()
            loss = model.get_loss(batch_torch, 'gen')
            loss.backward()
            optimizer_gen.step()
            loss_epoch += loss.item()

        loss_epoch /= n_batch
        toc_epoch = time.time() - tic

        loss_train[i] = loss_epoch
        toc_train[i] = toc_epoch
        if i in ckpts:
            pkl = {}
            pkl['loss_train'] = loss_train
            pkl['toc_train'] = toc_train
            pkl['model_state_dict'] = model.state_dict()
            pkl['optimizer_rec_state_dict'] = optimizer_rec.state_dict()
            pkl['optimizer_dis_state_dict'] = optimizer_dis.state_dict()
            pkl['optimizer_gen_state_dict'] = optimizer_gen.state_dict()
            torch.save(pkl, model_path_i)

        print((f'epoch {i + 1}/{n_epoch}, '
               f'loss={loss_epoch:0.4f}, '
               f'time={toc_epoch:0.2f}.'))
    return model

