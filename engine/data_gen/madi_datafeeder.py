# -*- coding: utf-8 -*-
"""
@author: <blinded>
"""

import numpy as np
import scipy.signal
from .util import _normalize


class MadiDatafeeder:
    def __init__(self, data, batch_size, data_config_name, seed=666):
        n_dim = data.shape[1]
        data_len = data.shape[2]

        freq_sigma = np.zeros((1, 1, 257, ))
        x = np.arange(5, 257, dtype=int)
        y = 57 * np.exp(-x * 0.05)
        freq_sigma[0, 0, x] = y
        x = np.arange(1, 6, dtype=int)
        y = 172 * np.exp(-x * 0.28)
        freq_sigma[0, 0, x] = y

        self.n_dim = n_dim
        self.data_len = data_len
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        self.freq_sigma = freq_sigma

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

        freq_sigma = self.freq_sigma

        freq_real = self.rng.normal(size=(batch_size, n_dim, 257, ))
        freq_mu = self.rng.normal(size=(batch_size, n_dim, 257, )) * 0.1
        freq_real = freq_real * freq_sigma + freq_mu

        freq_imag = self.rng.normal(size=(batch_size, n_dim, 257, ))
        freq_mu = self.rng.normal(size=(batch_size, n_dim, 257, )) * 0.1
        freq_imag = freq_imag * freq_sigma + freq_mu

        freq = freq_real + 1j * freq_imag
        batch = np.fft.irfft(freq, axis=2)
        batch = scipy.signal.resample(
            batch, data_len, axis=2)
        batch = _normalize(batch)
        return batch

