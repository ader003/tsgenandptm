# -*- coding: utf-8 -*-
"""
@author: <blinded>
"""


import numpy as np
import scipy.signal
from .util import _normalize


class SidiDatafeeder:
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

        freq_real = np.zeros((batch_size, n_dim, 257, ))
        freq_imag = np.zeros((batch_size, n_dim, 257, ))
        for i in range(batch_size):
            for j in range(n_dim):
                x, y = self._get_x_y()
                freq_real[i, j, x] = y
                x, y = self._get_x_y()
                freq_imag[i, j, x] = y

        freq = freq_real + 1j * freq_imag
        batch = np.fft.irfft(freq, axis=2)
        batch = scipy.signal.resample(
            batch, data_len, axis=2)
        batch = _normalize(batch)
        return batch

    def _get_x_y(self):
        x_mu = 5.61
        x_sigma = 13.37

        x = -1
        while x < 0:
            x = self.rng.normal() * x_sigma + x_mu
        x = int(x)

        y_mu = 163.93
        y_sigma = 72.95

        y = self.rng.normal() * y_sigma + y_mu
        y = self.rng.choice([1.0, -1.0]) * y
        return x, y

