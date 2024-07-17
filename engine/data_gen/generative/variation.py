# -*- coding: utf-8 -*-
"""
@author: <blinded>
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


class VAENet(nn.Module):
    def __init__(self, in_dim, n_dim=64, n_layer=5):
        super(VAENet, self).__init__()
        self.in_dim = in_dim
        self.n_layer = n_layer

        in_dim_ = in_dim
        out_dim = n_dim
        self.conv_00 = nn.Conv1d(
            in_dim_, out_dim, 3, padding=1)

        self.down_layers = OrderedDict()
        for i in range(n_layer):
            in_dim_ = out_dim
            out_dim = out_dim * 2
            self.down_layers[f'block_{i:02d}'] = Block(
                in_dim_, out_dim, block_type='down')
            self.add_module(
                f'down_layer_{i:02d}', self.down_layers[f'block_{i:02d}'])

        self.linear_mu = nn.Linear(out_dim, out_dim)
        self.linear_logvar = nn.Linear(out_dim, out_dim)
        self.laten_dim = out_dim

        self.up_layers = OrderedDict()
        for i in range(n_layer):
            in_dim_ = out_dim
            out_dim = out_dim // 2
            self.up_layers[f'block_{i:02d}'] = Block(
                in_dim_, out_dim, block_type='up')
            self.add_module(
                f'up_layer_{i:02d}', self.up_layers[f'block_{i:02d}'])
        self.conv_01 = nn.Conv1d(
            out_dim, in_dim, 1)
        self.dummy = nn.Parameter(torch.empty(0))

    def encode(self, x):
        h = self.conv_00(x)
        for key in self.down_layers:
            h = self.down_layers[key](h)
        h_len = h.size()[2]
        h = nn.AdaptiveAvgPool1d(1)(h)
        h = h[:, :, 0]
        h_mu = self.linear_mu(h)
        h_logvar = self.linear_logvar(h)
        return h_mu, h_logvar, h_len

    def reparameterize(self, h_mu, h_logvar):
        sigma = torch.exp(0.5 * h_logvar)
        eps = torch.randn_like(sigma)
        return eps * sigma + h_mu

    def decode(self, h, h_len):
        h = torch.unsqueeze(h, 2)
        h = nn.Upsample(size=h_len)(h)
        for key in self.up_layers:
            h = self.up_layers[key](h)
        x_hat = self.conv_01(h)
        return x_hat

    def forward(self, x):
        h_mu, h_logvar, h_len = self.encode(x)
        z = self.reparameterize(h_mu, h_logvar)
        x_hat = self.decode(z, h_len)
        return x_hat, h_mu, h_logvar

    def get_loss(self, x, beta=0.001):
        x_hat, h_mu, h_logvar = self.forward(x)
        recons_loss = F.mse_loss(x_hat, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + h_logvar - h_mu**2 -
                                               h_logvar.exp(), dim=1),
                              dim=0)
        loss = recons_loss + beta * kld_loss
        return loss

    @torch.no_grad()
    def sample(self, n_sample, sample_len):
        device = self.dummy.device
        laten_dim = self.laten_dim
        h_len = sample_len // (2 ** self.n_layer)
        z = torch.randn(
            (n_sample, laten_dim), device=device)
        x_hat = self.decode(z, h_len)
        return x_hat


def variation_train(data, model_path, model_config, device):
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
    beta = float(model_config['beta'])
    feeder = BaseDatafeeder(data, batch_size)
    n_batch = feeder.n_batch

    model = VAENet(in_dim, n_dim=n_dim, n_layer=n_layer)
    model.to(device)
    model.train()

    ckpts = _get_checkpoint(ckpt_gap, n_epoch)
    start_epoch = _get_start_epoch(model_path, ckpts)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr)

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

            optimizer.load_state_dict(
                pkl['optimizer_state_dict'])

        tic = time.time()
        loss_epoch = 0
        for j in range(n_batch):
            optimizer.zero_grad()
            batch = feeder.get_batch()
            batch_torch = torch.from_numpy(batch)
            batch_torch = batch_torch.to(device, dtype=torch.float)
            loss = model.get_loss(batch_torch, beta=beta)
            loss.backward()
            optimizer.step()
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
            pkl['optimizer_state_dict'] = optimizer.state_dict()
            torch.save(pkl, model_path_i)

        print((f'epoch {i + 1}/{n_epoch}, '
               f'loss={loss_epoch:0.4f}, '
               f'time={toc_epoch:0.2f}.'))
    return model


# def _test():
#     model = VAENet(3)

#     x = 144
#     print()
#     x = torch.randn(10, 3, 512)
#     x_hat, h_mu, h_logvar = model.forward(x)
#     print(x_hat.shape, h_mu.shape, h_logvar.shape)


# if __name__ == '__main__':
#     _test()

