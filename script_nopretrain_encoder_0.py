# -*- coding: utf-8 -*-
"""
@author: <blinded>
"""


import os
import copy
import argparse
import pathlib
import torch
from random import shuffle
from engine.util import parse_config
from engine.util import get_agg_result
from engine.data_io import get_ucr_data_names
from engine.data_io import get_uea_data_names
from engine.data_io import load_ucr_dataset
from engine.data_io import load_uea_dataset
from engine.model import get_model
from engine.train_test import nn_train
from engine.train_test import nn_eval

# non-pretrain
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
    if not is_mlp:
        data_config['data']['data_dir'] = (
            '/san-ssapfs' + data_config['data']['data_dir'])

    method_config = os.path.join(
        '.', 'config', f'{method_name}.config')
    method_config = parse_config(method_config)

    dataset_names, load_dataset = None, None
    if data_config_name == "ucr_00":
        dataset_names = get_ucr_data_names()
        load_dataset = load_ucr_dataset
    elif data_config_name == "uea_00":
        dataset_names = get_uea_data_names()
        load_dataset = load_uea_dataset
    else:
        print("Unsupported data config requested.")
        quit()

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
        result_dir = os.path.join(
            '.', 'result', f'{data_config_name}_{dataset_name}')
        result_agg_dir = os.path.join(
            '.', 'result_agg', f'{data_config_name}_{dataset_name}')
        model_dir = os.path.join(
            '.', 'model', f'{data_config_name}_{dataset_name}')

        path = pathlib.Path(result_dir)
        path.mkdir(parents=True, exist_ok=True)
        path = pathlib.Path(result_agg_dir)
        path.mkdir(parents=True, exist_ok=True)
        path = pathlib.Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)

        result_path = os.path.join(
            result_dir, f'{method_name}_{fmt_str}.npz')
        result_agg_path = os.path.join(
            result_agg_dir, f'{method_name}.npz')
        model_path = os.path.join(
            model_dir, f'{method_name}_{fmt_str}.npz')
        if os.path.isfile(result_agg_path):
            continue

        print(model_path)
        dataset = load_dataset(dataset_name, data_config)
        method_config_ = copy.deepcopy(method_config)
        method_config_['in_dim'] = dataset['n_dim']
        method_config_['n_class'] = dataset['n_class']
        method_config_['data_len'] = dataset['data_len']
        method_config_['encoder']['in_dim'] = dataset['data_train'].shape[1]
        method_config_['encoder']['data_len'] = dataset['data_train'].shape[2]
        model = get_model(method_config_)
        nn_train(dataset, model, model_path, method_config_['train'], device) # this is 1nn
        print(result_path)
        nn_eval(dataset, model, model_path, result_path, method_config_['train'], device)
        get_agg_result(result_path, result_agg_path, method_config_['train'])


if __name__ == '__main__':
    main_wrapper()

