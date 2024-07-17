# -*- coding: utf-8 -*-
"""
@author: <blinded>
"""


import os
import numpy as np
import torch
from scipy import signal
from .generative.adversarial import GANNet
from .util import _normalize
from ..util import parse_config


class GANDatafeeder:
    def __init__(self, data, batch_size, data_config_name, dataset_name,
                 model_config_name, device=None):
        in_dim = data.shape[1]
        data_len = data.shape[2]

        model_config = os.path.join(
            '.', 'config', f'{model_config_name}.config')
        model_config = parse_config(model_config)
        model_config = model_config['model']

        n_dim = int(model_config['n_dim'])
        n_layer = int(model_config['n_layer'])
        n_epoch = int(model_config['n_epoch'])

        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

        if data_len < 32:
            sample_len = 32
        else:
            sample_len = 2 ** np.ceil(np.log(data_len) / np.log(2))
            sample_len = int(sample_len)

        model = GANNet(in_dim, sample_len, n_dim=n_dim, n_layer=n_layer)
        model_dir = os.path.join(
            '.', 'model_gen', f'{data_config_name}_{dataset_name}')
        model_path = model_path = os.path.join(
            model_dir, f'{model_config_name}_{n_epoch - 1:04d}.npz')
        pkl = torch.load(model_path, map_location='cpu')
        model.load_state_dict(
            pkl['model_state_dict'])
        model.to(device)
        model.eval()

        self.n_data = None
        if data_config_name == 'ucr_00':
            self.n_data = 1494 # average size of all train+test splits for all datasets, rounded up to nearest whole
        elif data_config_name == 'uea_00':
            self.n_data = 3398
        if self.n_data < data.shape[0]:
            self.n_data = data.shape[0] # the original pretraining data size is greater than the threshold; do not cripple the performance, and use the same # of samples

        self.data_len = data_len
        self.sample_len = sample_len
        self.batch_size = batch_size
        self.model = model


    def get_batch(self):
        batch_size = self.batch_size
        data_len = self.data_len
        model = self.model

        sample = model.sample(
            batch_size)
        sample = sample.cpu().detach().numpy()
        sample = signal.resample(
            sample, data_len, axis=2)
        sample = _normalize(sample)
        return sample

