# -*- coding: utf-8 -*-
"""
@author: <blinded>
"""

import os
from .util import write_config


def get_dataset(config_dir, ucr_dir, uea_dir):
    config_dict = {}
    config_dict['data'] = {}
    config_dict['data']['data_dir'] = ucr_dir
    config_dict['data']['max_len'] = 512
    config_dict['data']['seed'] = 666
    config_dict['data']['pretrain_frac'] = 0.5
    config_dict['data']['train_frac'] = 0.3
    config_dict['data']['valid_frac'] = 0.1
    config_dict['data']['test_frac'] = 0.1

    # config_path = os.path.join(
    #     '.', 'config', 'ucr_00.config')
    config_path = os.path.join(
        config_dir, 'ucr_00.config')
    write_config(config_dict, config_path)

    config_dict = {}
    config_dict['data'] = {}
    config_dict['data']['data_dir'] = uea_dir
    config_dict['data']['max_len'] = 512
    config_dict['data']['seed'] = 666
    config_dict['data']['pretrain_frac'] = 0.5
    config_dict['data']['train_frac'] = 0.3
    config_dict['data']['valid_frac'] = 0.1
    config_dict['data']['test_frac'] = 0.1

    # config_path = os.path.join(
        # '.', 'config', 'uea_00.config')
    config_path = os.path.join(
        config_dir, 'ucr_00.config')
    write_config(config_dict, config_path)

