# -*- coding: utf-8 -*-
"""
@author: <blinded>
"""


import os
import copy
import argparse
import pathlib
import numpy as np
import torch
from random import shuffle
from engine.util import parse_config
from engine.data_io import get_ucr_data_names
from engine.data_io import get_uea_data_names
from engine.data_io import load_ucr_dataset
from engine.data_io import load_uea_dataset
from engine.data_gen.generative import diffusion_train
from engine.data_gen.generative import variation_train
from engine.data_gen.generative import adversarial_train


def main_wrapper():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name',
                        default='ucr_00')
    parser.add_argument('--method_name')
    parser.add_argument('--dataset_order',
                        type=int, choices=[-1, 0, 1, ], default=1)
    parser.add_argument('--is_mlp', type=int, choices=[0, 1, ], default=1)

    args = parser.parse_args()
    data_name = args.data_name
    method_name = args.method_name
    dataset_order = args.dataset_order
    is_mlp = bool(args.is_mlp)

    main(data_name, method_name, dataset_order, is_mlp=is_mlp)


def main(data_config_name, method_name, dataset_order, is_mlp=True):
    data_config = os.path.join(
        '.', 'config', f'{data_config_name}.config')
    data_config = parse_config(data_config)
    if is_mlp:
        data_config['data']['data_dir'] = (
            data_config['data']['data_dir'][11:])

    method_config = os.path.join(
        '.', 'config', f'{method_name}.config')
    method_config = parse_config(method_config)

    if 'ucr' in data_config_name:
        get_data_names = get_ucr_data_names
        load_dataset = load_ucr_dataset
    elif 'uea' in data_config_name:
        get_data_names = get_uea_data_names
        load_dataset = load_uea_dataset

    model_config = method_config['model']
    if model_config['name'] == 'diffusion':
        train = diffusion_train
    elif model_config['name'] == 'variation':
        train = variation_train
    elif model_config['name'] == 'adversarial':
        train = adversarial_train

    dataset_names = get_data_names()
    if dataset_order == -1:
        dataset_names = dataset_names[::-1]
    elif dataset_order == 0:
        shuffle(dataset_names)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(device)

    fmt_str = '{0:04d}'
    for dataset_name in dataset_names:
        model_dir = os.path.join(
            '.', 'model', f'{data_config_name}_{dataset_name}')

        path = pathlib.Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)

        model_path = os.path.join(
            model_dir, f'{method_name}_{fmt_str}.npz')

        print(model_path)
        dataset = load_dataset(dataset_name, data_config)
        data = dataset['data_pretrain']
        model_config['in_dim'] = data.shape[1]
        model_config['in_len'] = data.shape[2]
        train(data, model_path, model_config, device)


if __name__ == '__main__':
    main_wrapper()

