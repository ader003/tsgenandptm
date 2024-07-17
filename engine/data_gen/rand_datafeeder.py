# -*- coding: utf-8 -*-
"""
@author: <blinded>
"""


import numpy as np
from .util import _normalize


class RandDatafeeder:
    def __init__(self, data, batch_size, data_config_name, seed=666):
        n_dim = data.shape[1]
        data_len = data.shape[2]

        self.n_dim = n_dim
        self.data_len = data_len
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        self.n_data = None
        if data_config_name == 'ucr_00':
            self.n_data = 1494 # average size of all train+test splits for all datasets, rounded up to nearest whole
        elif data_config_name == 'uea_00':
            self.n_data = 3398
        if self.n_data < data.shape[0]:
            self.n_data = data.shape[0] # the original pretraining data size is greater than the threshold; do not cripple the performance, and use the same # of samples

    def get_batch(self):
        batch_size = self.batch_size
        n_dim = self.n_dim
        data_len = self.data_len

        batch = self.rng.normal(size=(batch_size, n_dim, data_len, ))
        batch = np.cumsum(batch, axis=2)
        batch = _normalize(batch)
        return batch

