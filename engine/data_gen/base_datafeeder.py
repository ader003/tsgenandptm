# -*- coding: utf-8 -*-
"""
@author: <blinded>
"""


import copy
import numpy as np


class BaseDatafeeder:
    def __init__(self, data, batch_size, seed=666):
        n_data = data.shape[0]
        n_dim = data.shape[1]
        data_len = data.shape[2]

        n_batch = int(np.ceil(n_data / batch_size))
        self.data = data
        self.n_data = n_data
        self.n_dim = n_dim
        self.data_len = data_len
        self.batch_size = batch_size
        self.n_batch = n_batch
        self.current_batch = 0
        self.rng = np.random.default_rng(seed)
        self._new_order()

    def get_batch(self):
        if self.current_batch == self.n_batch:
            self.current_batch = 0
            self._new_order()
        current_batch = self.current_batch
        order = self.order

        batch_size = self.batch_size
        n_dim = self.n_dim
        data_len = self.data_len
        index_start = current_batch * batch_size
        index_end = (current_batch + 1) * batch_size
        index_batch = order[index_start:index_end]

        batch = copy.deepcopy(
            self.data[index_batch, :, :])
        self.current_batch += 1
        return batch

    def _new_order(self):
        n_data = self.n_data
        batch_size = self.batch_size
        order = self.rng.permutation(n_data)
        order = np.concatenate(
            (order, order[:batch_size], ), axis=0)
        self.order = order

