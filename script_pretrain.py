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
from engine.data_io import load_ucr_dataset, load_uea_dataset
from engine.data_io import get_ucr_data_names, get_uea_data_names
from engine.model import get_model
from engine.train_test import nn_pretrain
from engine.train_test import train_tune_svm_theta
from engine.util import get_agg_result
from engine.train_test import nn_train, nn_eval

from engine.data_gen import BaseDatafeeder, DistDatafeeder, MadiDatafeeder, RandDatafeeder, SidiDatafeeder, DiffDatafeeder, BVAEDatafeeder, GANDatafeeder


def main_wrapper():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name',
                        default='ucr_00')
    parser.add_argument('--method_name')
    parser.add_argument('--dataset_order',
                        type=int, choices=[-1, 0, 1, ], default=1)

    args = parser.parse_args()
    data_name = args.data_name
    method_name = args.method_name
    dataset_order = args.dataset_order

    main(data_name, method_name, dataset_order)


def select_datafeeder(feeder_type, data_pretrain, batch_size, data_config_name, dataset_name):
    datafeeeder = None
    # pretrain-base-0/1
    if feeder_type == "base":
        datafeeder = BaseDatafeeder(data_pretrain, batch_size)
    # pretrain-gen-0/1
    elif feeder_type == "dist":
        datafeeder = DistDatafeeder(data_pretrain, batch_size, data_config_name)
    elif feeder_type == "madi":
        datafeeder = MadiDatafeeder(data_pretrain, batch_size, data_config_name)
    elif feeder_type == "rand":
        datafeeder = RandDatafeeder(data_pretrain, batch_size, data_config_name)
    elif feeder_type == "sidi":
        datafeeder = SidiDatafeeder(data_pretrain, batch_size, data_config_name)
    # method name will match to its .config
    elif feeder_type == "bvae":
        datafeeder = BVAEDatafeeder(data_pretrain, batch_size, data_config_name, dataset_name, "gen_bvae_00")
    elif feeder_type == "gan":
        datafeeder = GANDatafeeder(data_pretrain, batch_size, data_config_name, dataset_name, "gen_gan_00")
    elif feeder_type == "diff":
        datafeeder = DiffDatafeeder(data_pretrain, batch_size, data_config_name, dataset_name, "gen_diff_00")
    else:
        print("Usupported data generator.")
        quit()
    return datafeeder


def main(data_config_name, method_name, dataset_order):
    data_config = os.path.join(
        '.', 'config', f'{data_config_name}.config')
    data_config = parse_config(data_config)

    method_config = os.path.join(
        '.', 'config', f'{method_name}.config')
    method_config = parse_config(method_config)
    feeder_type = method_config['datafeeder']['feeder_type']

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    fmt_str = '{0:04d}'

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

    # for dataset_name in dataset_names:
    for dataset_name in ["ItalyPowerDemand",]:
        result_dir = os.path.join(
            '.', 'result', f'{data_config_name}_{dataset_name}')
        result_agg_dir = os.path.join(
            '.', 'result_agg', f'{data_config_name}_{dataset_name}')
        model_dir = os.path.join(
            '.', 'model', f'{data_config_name}_{dataset_name}')
        model_finetune_dir = os.path.join(
            '.', 'model_finetune', f'{data_config_name}_{dataset_name}')
        
        path = pathlib.Path(result_dir)
        path.mkdir(parents=True, exist_ok=True)
        path = pathlib.Path(result_agg_dir)
        path.mkdir(parents=True, exist_ok=True)
        path = pathlib.Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)
        path = pathlib.Path(model_finetune_dir)
        path.mkdir(parents=True, exist_ok=True)

        dataset = load_dataset(dataset_name, data_config)
        data_pretrain = dataset['data_pretrain']
        batch_size = int(method_config['train']['batch_size'])
        method_config['in_dim'] = data_pretrain.shape[1]
        method_config['data_len'] = data_pretrain.shape[2]
        method_config['encoder']['in_dim'] = data_pretrain.shape[1]
        method_config['encoder']['data_len'] = data_pretrain.shape[2]
        model = get_model(method_config)
        
        datafeeder = select_datafeeder(feeder_type, data_pretrain, batch_size, data_config_name, dataset_name)

        result_path = os.path.join(
            result_dir, f'{method_name}_{fmt_str}.npz')
        result_agg_path = os.path.join(
            result_agg_dir, f'{method_name}.npz')
        model_ckpt_name = method_name[:7]+method_name[-4:] # cut out l/nl for MODE 0 experiments
        model_path = os.path.join(
            model_dir, f'{model_ckpt_name}_{fmt_str}.npz')
        model_finetune_path = os.path.join(model_finetune_dir, f'{model_ckpt_name}_{fmt_str}')
        
        if os.path.isfile(result_agg_path):
            print("{} already computed on {}.".format(method_name,dataset_name))
            continue
        print("Pretraining {} on {}...".format(method_name, dataset_name))
        nn_pretrain(datafeeder, model, model_path, method_config['train'], device)
        model.eval()
        model.to(device)

        print("Finetuning the pretrained model {} on {}...".format(method_name, dataset_name))
        # load last epoch checkpoint pretrained model that was just trained in nn_pretrain()
        pretrain_model_path = method_config[model.pretrain_name]['pre_train_model']
        pretrained_model_path = pretrain_model_path.replace("ucr_00_pretrain",data_config_name+"_"+dataset_name) # replace placeholder for dataset
        pretrained_model_ckpt = torch.load(pretrained_model_path)
        pretrained_model = get_model(method_config, encoder_only=True)
        pretrained_model.state_dict(pretrained_model_ckpt['model_state_dict'])
        nn_train(dataset, pretrained_model, model_finetune_path, method_config['train'], device)
        nn_eval(dataset, pretrained_model, model_finetune_path, result_path, method_config['train'], device)
            

        print("Performing model selection on {} on {}".format(method_name,dataset_name))
        get_agg_result(result_path, result_agg_path, method_config['train'])

        
    return


if __name__ == '__main__':
    main_wrapper()
