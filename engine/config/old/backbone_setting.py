# -*- coding: utf-8 -*-
"""
@author:  <blinded>
"""

import os

def get_backbone(config_dict, backbone):
    if backbone == "LSTM":
        config_dict['encoder'] = get_lst_setting()
    if backbone == "GRU":
        config_dict['encoder'] = get_gru_setting()
    if backbone == "ResNet":
        config_dict['encoder'] = get_r1d_setting()
    if backbone == "XFMR":
        config_dict['encoder'] = get_trf_setting_0()

    config_dict['backbone']['model_type'] = backbone
    return config_dict


# -*- coding: utf-8 -*-
"""
@author:  <blinded>
"""


def get_gru_setting():
    encoder = {}
    encoder['in_dim'] = 'None'
    encoder['out_dim'] = 128
    encoder['rnn_type'] = 'GRU'
    encoder['n_layer'] = 2
    encoder['n_dim'] = 64
    encoder['is_projector'] = 'False'
    encoder['project_norm'] = 'None'
    encoder['dropout'] = 0.0
    return encoder


def get_lst_setting():
    encoder = {}
    encoder['in_dim'] = 'None'
    encoder['out_dim'] = 128
    encoder['rnn_type'] = 'LSTM'
    encoder['n_layer'] = 2
    encoder['n_dim'] = 64
    encoder['is_projector'] = 'False'
    encoder['project_norm'] = 'None'
    encoder['dropout'] = 0.0
    return encoder


def get_r1d_setting():
    encoder = {}
    encoder['in_dim'] = 'None'
    encoder['out_dim'] = 128
    encoder['n_dim'] = 64
    encoder['block_type'] = 'alternative'
    encoder['norm'] = 'None'
    encoder['is_projector'] = 'False'
    encoder['project_norm'] = 'None'
    return encoder


def get_trf_setting_0():
    encoder = {}
    encoder['in_dim'] = 'None'
    encoder['out_dim'] = 128
    encoder['n_layer'] = 4
    encoder['n_dim'] = 64
    encoder['n_head'] = 8
    encoder['norm_first'] = 'True'
    encoder['is_pos'] = 'True'
    encoder['is_projector'] = 'False'
    encoder['project_norm'] = 'None'
    encoder['dropout'] = 0.0
    return encoder

