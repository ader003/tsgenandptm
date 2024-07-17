# -*- coding: utf-8 -*-
"""
@author: <blinded>
"""


import numpy as np


def _normalize(data):
    data_mu = np.mean(data, axis=2, keepdims=True)
    data_sigma = np.std(data, axis=2, keepdims=True)
    data_sigma[data_sigma <= 0] = 1
    data = (data - data_mu) / data_sigma
    return data


def _time_to_freq(data):
    freq = np.fft.rfft(data, axis=2)
    freq_real = np.real(freq)
    freq_imag = np.imag(freq)
    return freq_real, freq_imag

