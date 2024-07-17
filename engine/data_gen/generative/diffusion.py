# -*- coding: utf-8 -*-
"""
@author:  <blinded>
"""


import os
import time
import copy
import math
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
    def __init__(self, in_dim, out_dim, pe_dim, block_type='down'):
        super(Block, self).__init__()
        self.pos_mlp = nn.Conv1d(pe_dim, out_dim, 1)

        if block_type == 'up':
            self.conv_00 = nn.Conv1d(
                in_dim * 2, out_dim, 3, padding=1)
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

    def forward(self, x, t_emb):
        h = self.conv_00(x)
        h = nn.ReLU()(h)
        h = self.norm_00(h)

        t_emb = self.pos_mlp(t_emb)
        t_emb = nn.ReLU()(t_emb)
        h = h + t_emb

        h = self.conv_01(h)
        h = nn.ReLU()(h)
        h = self.norm_01(h)

        h = self.transform_00(h)
        return h


class PositionalEncoding(nn.Module):
    def __init__(self, n_dim, n_step):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(n_step)
        div_term = torch.exp(
            torch.arange(0, n_dim, 2) * (-math.log(10000.0) / n_dim))
        pos_emb = torch.zeros(n_step, n_dim)

        position = position.unsqueeze(0)
        div_term = div_term.unsqueeze(1)
        pos_emb[:, 0::2] = torch.sin(div_term * position).T
        pos_emb[:, 1::2] = torch.cos(div_term * position).T
        self.register_buffer('pos_emb', pos_emb, persistent=False)

    def forward(self, t):
        return self.pos_emb[t, :]


class TSDUNet(nn.Module):
    def __init__(self, in_dim, n_dim=64, n_layer=5, n_step=400,
                 beta_start=0.0001, beta_end=0.02):
        super(TSDUNet, self).__init__()
        self.in_dim = in_dim
        self.n_layer = n_layer
        self.n_step = n_step

        self.pos_emb = nn.Sequential(
            PositionalEncoding(n_dim, n_step),
            nn.Linear(n_dim, n_dim),
            nn.ReLU()
        )

        in_dim_ = in_dim
        out_dim = n_dim
        self.conv_00 = nn.Conv1d(
            in_dim_, out_dim, 3, padding=1)

        self.down_layers = OrderedDict()
        for i in range(n_layer):
            in_dim_ = out_dim
            out_dim = out_dim * 2
            self.down_layers[f'block_{i:02d}'] = Block(
                in_dim_, out_dim, n_dim, block_type='down')
            self.add_module(
                f'down_layer_{i:02d}', self.down_layers[f'block_{i:02d}'])

        self.up_layers = OrderedDict()
        for i in range(n_layer):
            in_dim_ = out_dim
            out_dim = out_dim // 2
            self.up_layers[f'block_{i:02d}'] = Block(
                in_dim_, out_dim, n_dim, block_type='up')
            self.add_module(
                f'up_layer_{i:02d}', self.up_layers[f'block_{i:02d}'])
        self.conv_01 = nn.Conv1d(
            out_dim, in_dim, 1)
        self.alphas = _get_alphas(
            beta_start, beta_end, n_step)
        self.dummy = nn.Parameter(torch.empty(0))

    def forward(self, x, t):
        n_layer = self.n_layer
        t_emb = self.pos_emb(t)
        t_emb = torch.unsqueeze(t_emb, 2)

        h = self.conv_00(x)
        residuals = []
        for key in self.down_layers:
            h = self.down_layers[key](h, t_emb)
            residuals.append(h)

        for key in self.up_layers:
            residual = residuals.pop()
            h = torch.cat((h, residual), dim=1)
            h = self.up_layers[key](h, t_emb)
        return self.conv_01(h)

    def get_loss(self, x, t):
        x_, noise = _diffuse_sample(x, t, self.alphas)
        noise_ = self.forward(x_, t)
        return F.l1_loss(noise, noise_)

    @torch.no_grad()
    def sample(self, n_sample, sample_len):
        device = self.dummy.device
        in_dim = self.in_dim
        n_step = self.n_step
        x = torch.randn(
            (n_sample, in_dim, sample_len), device=device)
        x = torch.clamp(x, -1.0, 1.0)
        t = torch.ones(n_sample) * n_step - 1
        t = t.to(device, dtype=torch.int64)
        noise = self.forward(x, t)

        beta = _get_alpha_from_t(
            self.alphas['beta'], t)
        sqrt_one_minus_alpha_cumprod = _get_alpha_from_t(
            self.alphas['sqrt_one_minus_alpha_cumprod'], t)
        sqrt_recip_alpha = _get_alpha_from_t(
            self.alphas['sqrt_recip_alpha'], t)

        # x_ = (sqrt_recip_alpha *
        #       (x - beta * noise / sqrt_one_minus_alpha_cumprod))
        x_ = (sqrt_recip_alpha *
            (x - noise * sqrt_one_minus_alpha_cumprod))
        return x_


def _get_alpha_from_t(alphas, t):
    device = t.device
    alphas = alphas.to(device)
    alphas_ = alphas[t]
    alphas_ = torch.unsqueeze(alphas_, 1)
    alphas_ = torch.unsqueeze(alphas_, 1)
    return alphas_


def _diffuse_sample(x, t, alphas):
    device = x.device
    noise = torch.randn_like(x, device=device)
    t = t.to(device)

    sqrt_alpha_cumprod = _get_alpha_from_t(
        alphas['sqrt_alpha_cumprod'], t)
    sqrt_one_minus_alpha_cumprod = _get_alpha_from_t(
        alphas['sqrt_one_minus_alpha_cumprod'], t)

    x_ = sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise
    return x_, noise


def _reverse_sample(x, t, noise, alphas):
    beta = _get_alpha_from_t(
        alphas['beta'], t)
    sqrt_one_minus_alpha_cumprod = _get_alpha_from_t(
        alphas['sqrt_one_minus_alpha_cumprod'], t)
    sqrt_recip_alpha = _get_alpha_from_t(
        alphas['sqrt_recip_alpha'], t)

    x_ = (sqrt_recip_alpha *
          (x - noise * sqrt_one_minus_alpha_cumprod))
    return x_


def _get_alphas(beta_start, beta_end, n_step):
    beta = torch.linspace(beta_start, beta_end, n_step)
    alpha = 1.0 - beta
    alpha_cumprod = torch.cumprod(alpha, axis=0)
    alpha_cumprod_prev = F.pad(
        alpha_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alpha = torch.sqrt(1.0 / alpha)
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)
    posterior_variance = (beta * (1. - alpha_cumprod_prev) /
                          (1. - alpha_cumprod))
    alphas = {}
    alphas['beta'] = beta
    alphas['sqrt_alpha_cumprod'] = sqrt_alpha_cumprod
    alphas['sqrt_one_minus_alpha_cumprod'] = sqrt_one_minus_alpha_cumprod
    alphas['sqrt_recip_alpha'] = sqrt_recip_alpha
    alphas['posterior_variance'] = posterior_variance
    return alphas


def diffusion_train(data, model_path, model_config, device):
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
    n_step = int(model_config['n_step'])
    beta_start = float(model_config['beta_start'])
    beta_end = float(model_config['beta_end'])

    ckpt_gap = int(model_config['ckpt_gap'])
    n_epoch = int(model_config['n_epoch'])
    batch_size = int(model_config['batch_size'])
    lr = float(model_config['lr'])
    feeder = BaseDatafeeder(data, batch_size)
    n_batch = feeder.n_batch

    model = TSDUNet(in_dim, n_dim=n_dim, n_layer=n_layer, n_step=n_step,
                    beta_start=beta_start, beta_end=beta_end)
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

            t = torch.randint(0, n_step, (batch.shape[0], ))
            t = t.to(device, dtype=torch.int64)
            loss = model.get_loss(batch_torch, t)
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
#     model = TSDUNet(3)

#     x = 144
#     print()
#     x = torch.randn(10, 3, 512)
#     t = torch.randint(400, size=(10, ))

#     y = model.forward(x, t)
#     print(y.shape)

# if __name__ == '__main__':
#     _test()

