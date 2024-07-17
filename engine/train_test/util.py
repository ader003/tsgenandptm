# -*- coding: utf-8 -*-
"""
@author: <blinded>
"""


import os
import numpy as np
import torch


def _normalize_dataset(data):
    data_mu = np.mean(data, axis=2, keepdims=True)
    data_sigma = np.std(data, axis=2, keepdims=True)
    data_sigma[data_sigma <= 0] = 1
    data = (data - data_mu) / data_sigma
    return data


def _get_checkpoint(n_ckpt, n_epoch):
    if n_ckpt >= n_epoch:
        ckpts = np.arange(n_epoch)
    else:
        ckpts = np.arange(1, n_ckpt + 1)
        ckpts = n_epoch * (ckpts / n_ckpt) - 1
    ckpts = ckpts.astype(int)
    ckpts_dict = {}
    for ckpt in ckpts:
        ckpts_dict[ckpt] = 0

    last_ckpt = n_epoch - 1
    if last_ckpt not in ckpts_dict:
        ckpts_dict[last_ckpt] = 0
    return ckpts_dict


def _get_start_epoch(model_path, ckpts):
    ckpt_list = [ckpt for ckpt in ckpts] + [0, ]
    ckpt_list = np.array(ckpt_list)
    order = np.sort(ckpt_list)
    ckpt_list = ckpt_list[order]

    start_epoch = 0
    for i in ckpt_list:
        model_path_i = model_path.format(i)
        if os.path.isfile(model_path_i):
            try:
                pkl = torch.load(model_path_i, map_location='cpu')
            except:
                print(f'{model_path_i} can not be opened. It is removed!')
                os.remove(model_path_i)

        # if not os.path.isfile(model_path_i):
        #    start_epoch = i
        #    print("{} missing, setting {} as starting epoch.".format(model_path_i,i))
    return start_epoch


def _get_data(dataset):
    data_train = dataset['data_train']
    label_train = dataset['label_train']
    data_valid = dataset['data_valid']
    label_valid = dataset['label_valid']
    data_test = dataset['data_test']
    label_test = dataset['label_test']

    data_train = _normalize_dataset(data_train)
    data_valid = _normalize_dataset(data_valid)
    data_test = _normalize_dataset(data_test)

    return data_train, label_train, data_valid, label_valid, data_test, label_test

def _get_embeddings(data, model, batch_size): # train data
    n_data = data.shape[0]
    n_iter = np.ceil(n_data / batch_size)
    n_iter = int(n_iter)
    embeddings = np.zeros((n_data,model.out_dim))
    for i in range(n_iter):
        idx_start = i * batch_size
        idx_end = (i + 1) * batch_size
        if idx_end > n_data:
            idx_end = n_data

        data_batch = data[idx_start:idx_end, :, :]
        embeddings[idx_start:idx_end,:] = model.encode(data_batch, normalize=False, to_numpy=True)
    return np.asarray(embeddings) 

