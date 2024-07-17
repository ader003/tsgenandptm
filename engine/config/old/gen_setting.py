# -*- coding: utf-8 -*-
"""
@author:  <blinded>
"""


import os
from .util import write_config


def get_generative():
    config_dict = {}
    config_dict['model'] = {}
    config_dict['model']['name'] = 'diffusion'
    config_dict['model']['n_dim'] = 64
    config_dict['model']['n_layer'] = 5
    config_dict['model']['n_step'] = 400
    config_dict['model']['beta_start'] = 0.0001
    config_dict['model']['beta_end'] = 0.02
    config_dict['model']['ckpt_gap'] = 8
    config_dict['model']['n_epoch'] = 400
    config_dict['model']['batch_size'] = 32
    config_dict['model']['lr'] = 0.0001

    config_path = os.path.join(
        '.', 'config', 'gen_diff_00.config')
    write_config(config_dict, config_path)

    config_dict = {}
    config_dict['model'] = {}
    config_dict['model']['name'] = 'variation'
    config_dict['model']['n_dim'] = 64
    config_dict['model']['n_layer'] = 5
    config_dict['model']['beta'] = 0.001
    config_dict['model']['ckpt_gap'] = 8
    config_dict['model']['n_epoch'] = 400
    config_dict['model']['batch_size'] = 32
    config_dict['model']['lr'] = 0.0001

    config_path = os.path.join(
        '.', 'config', 'gen_bvae_00.config')
    write_config(config_dict, config_path)

    config_dict = {}
    config_dict['model'] = {}
    config_dict['model']['name'] = 'adversarial'
    config_dict['model']['n_dim'] = 64
    config_dict['model']['n_layer'] = 5
    config_dict['model']['ckpt_gap'] = 8
    config_dict['model']['n_epoch'] = 400
    config_dict['model']['batch_size'] = 32
    config_dict['model']['lr'] = 0.0001

    config_path = os.path.join(
        '.', 'config', 'gen_gan_00.config')
    write_config(config_dict, config_path)

