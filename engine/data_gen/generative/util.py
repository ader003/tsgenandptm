# -*- coding: utf-8 -*-
"""
@author:  <blinded>
"""


import os
import numpy as np
import torch


def _get_checkpoint(ckpt_gap, n_epoch):
    if ckpt_gap >= n_epoch:
        ckpts = np.arange(n_epoch)
    else:
        ckpts = np.arange(ckpt_gap, n_epoch, ckpt_gap)
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
    order = np.argsort(-ckpt_list)
    ckpt_list = ckpt_list[order]

    for i in ckpt_list:
        model_path_i = model_path.format(i)
        if os.path.isfile(model_path_i):
            try:
                pkl = torch.load(model_path_i, map_location='cpu')
            except:
                print(f'{model_path_i} can not be opened. It is removed!')
                os.remove(model_path_i)

    for i in ckpt_list:
        model_path_i = model_path.format(i)
        if os.path.isfile(model_path_i):
            return i
    return 0

