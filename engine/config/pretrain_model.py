# -*- coding: utf-8 -*-
"""
@author: <blinded>
"""


import os
from .base_setting import get_r1d_setting
from .base_setting import get_trf_setting_0
from .base_setting import get_timefreq_setting
from .base_setting import get_ts2vec_setting
from .base_setting import get_mixup_setting
from .base_setting import get_simclr_setting
from .base_setting import get_timeclr_setting
from .base_setting import get_classifier_setting
from .base_setting import get_train_setting
from .util import write_config
import numpy as np

def _get_pretrain_setting(config_dir, short_name, setting_fun, setting_str,
                          pretrain_setting, datafeeder):
    prefix = f'{short_name}_{pretrain_setting[0]}'
    model_name = f'{pretrain_setting[1]}_{setting_str}'

    batch_size = 256
    norm = 'LN'
    config_dict = {}
    config_dict['model'] = {'model_name': model_name}

    feeder_map = {0: "base",
                  1: "bvae",
                  2: "gan",
                  3: "diff",
                  4: "dist",
                  5: "madi",
                  6: "rand",
                  7: "sidi",
                  }
    config_dict['datafeeder'] = {'feeder_type': feeder_map[datafeeder]}

    config_dict['encoder'] = setting_fun()
    if 'norm' in config_dict['encoder']:
        config_dict['encoder']['norm'] = norm
    config_dict['encoder']['in_dim'] = 1

    if pretrain_setting[1] == 'timefreq':
        if 'out_dim' in config_dict['encoder']:
            config_dict['encoder']['out_dim'] = int(
                config_dict['encoder']['out_dim'] / 2)
        if 'n_dim' in config_dict['encoder']:
            config_dict['encoder']['n_dim'] = int(
                config_dict['encoder']['n_dim'] / 2)
        config_dict['timefreq'] = get_timefreq_setting()
        config_dict['timefreq']['project_norm'] = norm

    elif pretrain_setting[1] == 'ts2vec':
        config_dict['ts2vec'] = get_ts2vec_setting()
        config_dict['encoder']['is_projector'] = 'True'
        config_dict['encoder']['project_norm'] = norm

    elif pretrain_setting[1] == 'mixup':
        config_dict['mixup'] = get_mixup_setting()
        config_dict['encoder']['is_projector'] = 'True'
        config_dict['encoder']['project_norm'] = norm

    elif pretrain_setting[1] == 'simclr':
        config_dict['simclr'] = get_simclr_setting()
        config_dict['encoder']['is_projector'] = 'True'
        config_dict['encoder']['project_norm'] = norm

    elif pretrain_setting[1] == 'timeclr':
        config_dict['timeclr'] = get_timeclr_setting()
        config_dict['encoder']['is_projector'] = 'True'
        config_dict['encoder']['project_norm'] = norm

    config_dict['train'] = get_train_setting()
    config_dict['train']['n_ckpt'] = 400
    config_dict['train']['batch_size'] = batch_size

    config_path = os.path.join(
        config_dir, f'{prefix}_{datafeeder:04d}.config')
    # config_wrapper = {'config': config_dict, 'config_type': 'legacy'}
    print(config_path)
    write_config(config_dict, config_path)


def _get_classifier_setting(config_dir, short_name, setting_fun, setting_str,
                            pretrain_setting, datafeeder, add_process=None):
    if add_process is None:
        prefix = f'{short_name}_{pretrain_setting[0]}_'
    else:
        prefix = f'{short_name}_{pretrain_setting[0]}_{add_process}_'

    model_name = f'{pretrain_setting[1]}_{setting_str}'
    batch_size = 256
    norm = 'LN'
    pretrain_data = 'ucr_00_pretrain'
    config_dict = {}
    config_dict['model'] = {'model_name': model_name}

    if add_process is not None:
        config_dict['svm'] = {}
        if add_process == 'l':
            config_dict['svm']['c_type'] = 'linear'
        elif add_process == 'nl':
            config_dict['svm']['c_type'] = 'nonlinear'

    feeder_map = {0: "base",
                  1: "bvae",
                  2: "gan",
                  3: "diff",
                  4: "dist",
                  5: "madi",
                  6: "rand",
                  7: "sidi",
                  }
    config_dict['datafeeder'] = {'feeder_type': feeder_map[datafeeder]}

    config_dict['encoder'] = setting_fun()
    if 'norm' in config_dict['encoder']:
        config_dict['encoder']['norm'] = norm
    config_dict['encoder']['in_dim'] = 1

    if pretrain_setting[1] == 'timefreq':
        if 'out_dim' in config_dict['encoder']:
            config_dict['encoder']['out_dim'] = int(
                config_dict['encoder']['out_dim'] / 2)
        if 'n_dim' in config_dict['encoder']:
            config_dict['encoder']['n_dim'] = int(
                config_dict['encoder']['n_dim'] / 2)
        config_dict['timefreq'] = get_timefreq_setting()
        config_dict['timefreq']['project_norm'] = norm
        config_dict['timefreq']['pre_train_model'] = (
            os.path.join(
                '.', 'model', pretrain_data,
                f'{short_name}_tf_{datafeeder:04d}_0399.npz'))

    elif pretrain_setting[1] == 'ts2vec':
        config_dict['ts2vec'] = get_ts2vec_setting()
        config_dict['encoder']['is_projector'] = 'True'
        config_dict['encoder']['project_norm'] = norm
        config_dict['ts2vec']['pre_train_model'] = (
            os.path.join(
                '.', 'model', pretrain_data,
                f'{short_name}_tv_{datafeeder:04d}_0399.npz'))

    elif pretrain_setting[1] == 'mixup':
        config_dict['mixup'] = get_mixup_setting()
        config_dict['encoder']['is_projector'] = 'True'
        config_dict['encoder']['project_norm'] = norm
        config_dict['mixup']['pre_train_model'] = (
            os.path.join(
                '.', 'model', pretrain_data,
                f'{short_name}_mu_{datafeeder:04d}_0399.npz'))

    elif pretrain_setting[1] == 'timeclr':
        config_dict['timeclr'] = get_timeclr_setting()
        config_dict['encoder']['is_projector'] = 'True'
        config_dict['encoder']['project_norm'] = norm
        config_dict['timeclr']['pre_train_model'] = (
            os.path.join(
                '.', 'model', pretrain_data,
                f'{short_name}_tc_{datafeeder:04d}_0399.npz'))

    config_dict['train'] = get_train_setting()
    config_path = os.path.join(
        config_dir, f'{prefix}{datafeeder:04d}.config')
    
    print(config_path)
    write_config(config_dict, config_path)


def get_pretrain_model(config_dir):
    pretrain_settings = [
        ['tf', 'timefreq', ],
        ['tv', 'ts2vec', ],
        ['mu', 'mixup', ],
        ['tc', 'timeclr', ],
    ]

    for datafeeder in np.arange(0,8):
        for pretrain_setting in pretrain_settings:
            _get_classifier_setting(
                config_dir, 'r1d', get_r1d_setting,
                'resnet1d', pretrain_setting, datafeeder)
            _get_classifier_setting(
                config_dir, 'trf', get_trf_setting_0,
                'transform', pretrain_setting, datafeeder)

