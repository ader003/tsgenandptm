# -*- coding: utf-8 -*-
"""
@author: <blinded>
"""


import numpy as np
from .util import _normalize
from .util import _time_to_freq


class DistDatafeeder:
    def __init__(self, data, batch_size, data_config_name, seed=666):
        n_dim = data.shape[1]
        data_len = data.shape[2]

        freq_real, freq_imag = _time_to_freq(data)
        freq_len = freq_real.shape[2]
        freq_real_mu = np.zeros((n_dim, freq_len, ))
        freq_imag_mu = np.zeros((n_dim, freq_len, ))
        freq_real_sigma = np.zeros((n_dim, freq_len, ))
        freq_imag_sigma = np.zeros((n_dim, freq_len, ))
        for i in range(n_dim):
            freq_real_mu[i, :] = np.mean(freq_real[:, i, :], axis=0)
            freq_imag_mu[i, :] = np.mean(freq_imag[:, i, :], axis=0)
            freq_real_sigma[i, :] = np.std(freq_real[:, i, :], axis=0)
            freq_imag_sigma[i, :] = np.std(freq_imag[:, i, :], axis=0)

        self.n_dim = n_dim
        self.data_len = data_len
        self.freq_len = freq_len
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        self.freq_real_mu = freq_real_mu
        self.freq_imag_mu = freq_imag_mu
        self.freq_real_sigma = freq_real_sigma
        self.freq_imag_sigma = freq_imag_sigma

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
        freq_len = self.freq_len

        freq_real_mu = self.freq_real_mu
        freq_imag_mu = self.freq_imag_mu
        freq_real_sigma = self.freq_real_sigma
        freq_imag_sigma = self.freq_imag_sigma

        batch = np.zeros((batch_size, n_dim, data_len, ))
        for i in range(n_dim):
            freq_real = self.rng.normal(size=(batch_size, freq_len, ))
            freq_real = (freq_real * freq_real_sigma[i, :] +
                         freq_real_mu[i, :])

            freq_imag = self.rng.normal(size=(batch_size, freq_len, ))
            freq_imag = (freq_imag * freq_imag_sigma[i, :] +
                         freq_imag_mu[i, :])

            freq = freq_real + 1j * freq_imag
            tmp = np.fft.irfft(freq, axis=1)
#            print(batch[:,i,:tmp.shape[1]].shape)
#            print(tmp.shape)
            batch[:, i, :tmp.shape[1]] = tmp
        batch = _normalize(batch)
        return batch

