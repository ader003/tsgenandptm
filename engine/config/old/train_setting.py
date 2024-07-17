# -*- coding: utf-8 -*-
"""
@author:  <blinded>
"""

def get_train_setting():
    train = {}
    train['lr'] = 0.0001
    train['batch_size'] = 64
    train['n_epoch'] = 400
    train['n_ckpt'] = 400
    return train
