# -*- coding: utf-8 -*-
"""
@author:  <blinded>
"""

import os

uae_dir = None # replace
ucr_dir = None # replace


def get_dataset(config_dict, config_path, dataset):
    config_dict['data'] = {}
    if dataset == "ucr":
        config_dict['data']['data_dir'] = ucr_dir
        config_dict['data']['max_len'] = 512
        config_dict['data']['seed'] = 666
        config_dict['data']['pretrain_frac'] = 0.5
        config_dict['data']['train_frac'] = 0.3
        config_dict['data']['valid_frac'] = 0.1
        config_dict['data']['test_frac'] = 0.1

        config_path = os.path.join(
            '.', 'config', 'ucr_')

    if dataset == "uea":
        config_dict['data']['data_dir'] = uae_dir
        config_dict['data']['max_len'] = 512
        config_dict['data']['seed'] = 666
        config_dict['data']['pretrain_frac'] = 0.5
        config_dict['data']['train_frac'] = 0.3
        config_dict['data']['valid_frac'] = 0.1
        config_dict['data']['test_frac'] = 0.1

        config_path = os.path.join(
            '.', 'config', 'uea_')
    return config_dict, config_path
