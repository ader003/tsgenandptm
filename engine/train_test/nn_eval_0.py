# -*- coding: utf-8 -*-
"""
@author:  <blinded>
"""

import os
import time
import random
import numpy as np
import torch
from .util import _get_checkpoint
from .util import _normalize_dataset
from sklearn.svm import SVC
from .util import _get_embeddings, _get_data


def _train_predict_svm(dataset, model, train_config, c_type='nonlinear'):
    # load and normalize data
    data_train, label_train, data_valid, label_valid, data_test, label_test = _get_data(dataset)

    # select svm type
    if c_type == 'linear':
        clf = SVC(kernel='linear')
    elif c_type == 'nonlinear':
        clf = SVC(kernel='rbf', gamma='auto')
    batch_size = int(train_config['batch_size'])

    # 1. fit svm to train embeddings
    # 2. score validation and test set embeddings
    tic = time.time()
    embeddings_train = _get_embeddings(data_train,model,batch_size)
    clf.fit(embeddings_train,label_train)

    embeddings_valid = _get_embeddings(data_valid,model,batch_size)
    predict_valid = clf.predict(embeddings_valid)
    acc_valid = clf.score(embeddings_valid, label_valid)
    
    embeddings_test = _get_embeddings(data_test,model,batch_size)
    predict_test = clf.predict(embeddings_test)
    acc_test = clf.score(embeddings_test, label_test)

    time_elapsed = time.time() - tic
    return predict_valid, acc_valid, predict_test, acc_test, time_elapsed


def svm_eval(dataset, model, model_path, result_path, train_config, device):
    label_valid = dataset['label_valid']
    label_test = dataset['label_test']
    n_epoch = int(train_config['n_epoch'])
    n_ckpt = int(train_config['n_ckpt'])
    ckpts = _get_checkpoint(n_ckpt, n_epoch)
    ckpts = [ckpt for ckpt in ckpts]
    ckpts = ckpts[::-1]
    random.shuffle(ckpts)

    for i in ckpts:
        result_path_i = result_path.format(i)
        if not os.path.isfile(result_path_i):
            continue
        try:
            result = np.load(result_path_i, allow_pickle=True)
        except:
            print(f'{result_path_i} can not be opened. It is removed!')
            os.remove(result_path_i)

    for i in ckpts:
        model_path_i = model_path.format(i)
        result_path_i = result_path.format(i)
        if os.path.isfile(result_path_i):
            result = np.load(result_path_i, allow_pickle=True)
            acc_valid_l = result['acc_valid_l']
            acc_test_l = result['acc_test_l']
            time_elapsed_l = result['time_elapsed_l']
            acc_valid_nl = result['acc_valid_nl']
            acc_test_nl = result['acc_test_nl']
            time_elapsed_nl = result['time_elapsed_nl']
            print((f'{result_path_i}, {acc_valid_l:0.4f}, {acc_valid_nl:0.4f}, {acc_test_l:0.4f}, {acc_test_nl:0.4f}, {time_elapsed_l:0.4f}, {time_elapsed_nl:0.4f}'))
            continue

        pkl = torch.load(model_path_i, map_location='cpu')
        model.load_state_dict( 
            pkl['model_state_dict'])
        model.to(device)
        model.eval()

        predict_valid_l, acc_valid_l, predict_test_l, acc_test_l, time_elapsed_l = _train_predict_svm(dataset, model, train_config, c_type='linear')
        
        predict_valid_nl, acc_valid_nl, predict_test_nl, acc_test_nl, time_elapsed_nl = _train_predict_svm(dataset, model, train_config, c_type='nonlinear')

        np.savez(result_path_i,
                 label_valid=label_valid,
                 label_test=label_test,
                 predict_valid_l=predict_valid_l,
                 predict_test_l=predict_test_l,
                 predict_valid_nl=predict_valid_nl,
                 predict_test_nl=predict_test_nl,
                 acc_valid_l=acc_valid_l,
                 acc_test_l=acc_test_l,
                 acc_valid_nl=acc_valid_nl,
                 acc_test_nl=acc_test_nl,
                 time_elapsed_l=time_elapsed_l,
                 time_elapsed_nl=time_elapsed_nl)
        print((f'{result_path_i}, {acc_valid_l:0.4f}, {acc_valid_nl:0.4f}, {acc_test_l:0.4f}, {acc_test_nl:0.4f}, {time_elapsed_l:0.4f}, {time_elapsed_nl:0.4f}'))

############################################################################################################################################
def _get_predict(data, label, model, train_config):
    n_data = data.shape[0]
    batch_size = int(train_config['batch_size'])
    n_iter = np.ceil(n_data / batch_size)
    n_iter = int(n_iter)

    tic = time.time()
    predict = np.zeros(n_data, dtype=int)
    for i in range(n_iter):
        idx_start = i * batch_size
        idx_end = (i + 1) * batch_size
        if idx_end > n_data:
            idx_end = n_data

        data_batch = data[idx_start:idx_end, :, :]
        logit = model.forward(
            data_batch, normalize=False, to_numpy=True)
        predict[idx_start:idx_end] = np.argmax(logit, axis=1)
    predict_time = time.time() - tic
    acc = np.sum(predict == label) / n_data
    return predict, acc, predict_time


def nn_eval(dataset, model, model_path, result_path, train_config, device):
    data_valid = dataset['data_valid']
    data_test = dataset['data_test']

    label_valid = dataset['label_valid']
    label_test = dataset['label_test']

    data_valid = _normalize_dataset(data_valid)
    data_test = _normalize_dataset(data_test)

    n_epoch = int(train_config['n_epoch'])
    n_ckpt = int(train_config['n_ckpt'])
    ckpts = _get_checkpoint(n_ckpt, n_epoch)
    ckpts = [ckpt for ckpt in ckpts]
    ckpts = ckpts[::-1]
    random.shuffle(ckpts)

    for i in ckpts:
        result_path_i = result_path.format(i)
        if not os.path.isfile(result_path_i):
            continue
        try:
            result = np.load(result_path_i, allow_pickle=True)
        except:
            print(f'{result_path_i} can not be opened. It is removed!')
            os.remove(result_path_i)

    for i in ckpts:
        model_path_i = model_path.format(i)
        result_path_i = result_path.format(i)
        if os.path.isfile(result_path_i):
            result = np.load(result_path_i, allow_pickle=True)
            acc_valid = result['acc_valid']
            acc_test = result['acc_test']
            time_valid = result['time_valid']
            time_test = result['time_test']
            print((f'{result_path_i}, {acc_valid:0.4f}, {acc_test:0.4f}, '
                  f'{time_valid+time_test:0.2f}'))
            continue

        pkl = torch.load(model_path_i, map_location='cpu')
        model.load_state_dict( 
            pkl['model_state_dict'])
        model.to(device)
        model.eval() 

        predict_valid, acc_valid, time_valid = _get_predict(
            data_valid, label_valid, model, train_config)
        predict_test, acc_test, time_test = _get_predict(
            data_test, label_test, model, train_config)

        np.savez(result_path_i,
                 label_valid=label_valid,
                 label_test=label_test,
                 predict_valid=predict_valid,
                 predict_test=predict_test,
                 acc_valid=acc_valid,
                 acc_test=acc_test,
                 time_valid=time_valid,
                 time_test=time_test)
        print((f'{result_path_i}, {acc_valid:0.4f}, {acc_test:0.4f}, '
               f'{time_valid+time_test:0.2f}'))

