# -*- coding: utf-8 -*-
"""
@author:  <blinded>
"""


import os
import time
import copy
import numpy as np
import scipy.spatial
import torch
import torch.nn as nn
from .util import _get_checkpoint
from .util import _normalize_dataset
from .loss import NTXentLossPoly
from .loss import NTXentLoss
from .loss import HierContrastLoss
from .loss import MixupLoss


def _get_start_epoch(model_path, ckpts):
    start_epoch = 0
    ckpts = np.sort(list(ckpts.keys()))
    for i in ckpts:
        model_path_i = model_path.format(i)
        if os.path.isfile(model_path_i):
            start_epoch = i
        else:
            break

    model_path_i = model_path.format(start_epoch) 
    if not os.path.isfile(model_path_i):
        return start_epoch

    try:
        pkl = torch.load(model_path_i, map_location='cpu')
    except:
        print(f'{model_path_i} can not be opened. It is removed!')
        os.remove(model_path_i)
        start_epoch = _get_start_epoch(model_path, ckpts)
    return start_epoch 


def _timefreq_encoder_forward(model, data_batch):
    h_t, z_t, h_f, z_f = model.forward(
        data_batch, normalize=False, to_numpy=False, is_augment=False)
    h_t_aug, z_t_aug, h_f_aug, z_f_aug = model.forward(
        data_batch, normalize=False, to_numpy=False, is_augment=True)
    loss_fun = NTXentLossPoly()
    loss_t = loss_fun(h_t, h_t_aug)
    loss_f = loss_fun(h_f, h_f_aug)
    loss_tf = loss_fun(z_t, z_f)
    loss = 0.2 * (loss_t + loss_f) + loss_tf
    return loss


def _timeclr_encoder_forward(model, data_batch):
    ts_emb_aug_0 = model.forward(
        data_batch, normalize=False, to_numpy=False, is_augment=True)
    ts_emb_aug_1 = model.forward(
        data_batch, normalize=False, to_numpy=False, is_augment=True)
    loss_fun = NTXentLossPoly()
    loss = loss_fun(ts_emb_aug_0, ts_emb_aug_1)
    return loss


def _ts2vec_encoder_forward(model, data_batch):
    ts_emb_l, ts_emb_r = model.forward(
        data_batch, normalize=False, to_numpy=False, is_augment=True)
    loss_fun = HierContrastLoss()
    loss = loss_fun(ts_emb_l, ts_emb_r)
    return loss


def _mixup_encoder_forward(model, data_batch):
    ts_emb_0, ts_emb_1, ts_emb_aug, lam = model.forward(
        data_batch, normalize=False, to_numpy=False, is_augment=True)
    loss_fun = MixupLoss()
    loss = loss_fun(ts_emb_0, ts_emb_1, ts_emb_aug, lam)
    return loss


def _mixclr_encoder_forward(model, data_batch):
    version = model.version
    if version == 0:
        ts_emb_0, ts_emb_1, ts_emb_aug, lam = model.forward(
            data_batch, normalize=False, to_numpy=False, is_augment=True)
        loss_fun = MixupLoss()
        loss = loss_fun(ts_emb_0, ts_emb_1, ts_emb_aug, lam)
    elif version == 1:
        (ts_emb_0, ts_emb_1, ts_emb_2, ts_emb_3,
         ts_emb_aug, lam) = model.forward(
            data_batch, normalize=False, to_numpy=False, is_augment=True)
        loss_fun = MixupLoss()
        loss_0 = loss_fun(ts_emb_0, ts_emb_1, ts_emb_aug, lam)

        loss_fun = NTXentLossPoly()
        loss_1 = loss_fun(ts_emb_2, ts_emb_3)
        loss = loss_0 + loss_1
    return loss


def nn_pretrain(datafeeder, model, model_path, train_config, device):
    model.to(device)
    model.train()

    pretrain_name = model.pretrain_name
    lr = float(train_config['lr'])

    n_data = datafeeder.n_data  # DATAFEEDER
    batch_size = int(train_config['batch_size'])
    n_iter = np.ceil(n_data / batch_size)
    n_iter = int(n_iter)
    n_epoch = int(train_config['n_epoch'])
    n_ckpt = int(train_config['n_ckpt'])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, lr * 100, epochs=n_epoch, steps_per_epoch=n_iter)

    ckpts = _get_checkpoint(n_ckpt, n_epoch)
    start_epoch = _get_start_epoch(model_path, ckpts)
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

            print((f'epoch {i + 1}/{n_epoch}, '
                   f'loss={loss_epoch:0.4f}, '
                   f'time={toc_epoch:0.2f}.'))
            continue

        model_state_dict_old = copy.deepcopy(
            model.state_dict())
        optimizer_state_dict_old = copy.deepcopy(
            optimizer.state_dict())

        tic = time.time()
        loss_epoch = 0
        for j in range(n_iter):
            data = datafeeder.get_batch()  # DATAFEEDER
            optimizer.zero_grad()
            pretrain_name = model.pretrain_name
            if pretrain_name == 'timefreq':
                loss = _timefreq_encoder_forward(model, data)
            elif pretrain_name == 'ts2vec':
                loss = _ts2vec_encoder_forward(model, data)
            elif pretrain_name == 'mixup':
                loss = _mixup_encoder_forward(model, data)
            elif pretrain_name == 'timeclr':
                loss = _timeclr_encoder_forward(model, data)
            elif pretrain_name == 'mixclr':
                loss = _mixclr_encoder_forward(model, data)
            else:
                raise Exception(
                    f'unknown pretrain name: {pretrain_name}')
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_epoch += loss.item()

        loss_epoch /= n_iter
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

