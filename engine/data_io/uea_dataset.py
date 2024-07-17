# -*- coding: utf-8 -*-
"""
@author: <blinded>
"""


import os
import numpy as np
from scipy import signal
from .util import _relabel


def load_uea_dataset(data_name, data_config):
    data_config = data_config['data']
    data_dir = data_config['data_dir']
    max_len = int(data_config['max_len'])
    seed = int(data_config['seed'])
    pretrain_frac = float(data_config['pretrain_frac'])
    train_frac = float(data_config['train_frac'])
    valid_frac = float(data_config['valid_frac'])
    test_frac = float(data_config['test_frac'])
    # assert pretrain_frac + train_frac + valid_frac + test_frac == 1.0

    data_path = os.path.join(
        data_dir, data_name + '.npz')
    dataset = np.load(data_path)

    data_test = dataset['data_test']
    label_test = dataset['label_test']
    data_train = dataset['data_train']
    label_train = dataset['label_train']

    n_test = label_test.shape[0]
    n_train = label_train.shape[0]
    test_len = data_test.shape[2]
    train_len = data_train.shape[2]

    if test_len != train_len:
        n_dim = data_test.shape[1]
        new_len = max(test_len, train_len)
        data_test_ = np.zeros((n_test, n_dim, new_len, ))
        data_train_ = np.zeros((n_train, n_dim, new_len, ))
        data_test_[:, :, :test_len] = data_test
        data_train_[:, :, :train_len] = data_train
        data_test = data_test_
        data_train = data_train_

    data_len = data_test.shape[2]
    if data_len > max_len:
        data_test = signal.resample(
            data_test, max_len, axis=2)
        data_train = signal.resample(
            data_train, max_len, axis=2)

    data = np.concatenate(
        (data_test, data_train, ), axis=0)
    label = np.concatenate(
        (label_test, label_train, ), axis=0)
    n_data = data.shape[0]
    n_dim = data.shape[1]
    data_len = data.shape[2]

    rng = np.random.default_rng(seed=seed)
    random_vec = rng.permutation(n_data)
    data = data[random_vec, :, :]
    label = label[random_vec]

    label, n_class = _relabel(label)
    if np.isclose(pretrain_frac, 1.0):
        data_pretrain = data
        data_train = None
        data_valid = None
        data_test = None
        label_train = None
        label_valid = None
        label_test = None
    else:
        data_pretrain = []
        data_train = []
        data_valid = []
        data_test = []
        label_train = []
        label_valid = []
        label_test = []
        for i in range(n_class):
            data_i = data[label == i, :, :]
            label_i = label[label == i]

            n_data_i = label_i.shape[0]
            n_train_i = np.round(train_frac * n_data_i)
            n_train_i = int(n_train_i)
            n_train_i = max(n_train_i, 1)

            n_valid_i = np.round(valid_frac * n_data_i)
            n_valid_i = int(n_valid_i)
            n_valid_i = max(n_valid_i, 1)

            n_test_i = np.round(test_frac * n_data_i)
            n_test_i = int(n_test_i)
            n_test_i = max(n_test_i, 1)

            n_pretrain_i = n_data_i - n_train_i - n_valid_i - n_test_i

            train_start = 0
            train_end = n_train_i

            valid_start = train_end
            valid_end = valid_start + n_valid_i

            test_start = valid_end
            test_end = test_start + n_test_i

            pretrain_start = test_end
            pretrain_end = pretrain_start + n_pretrain_i

            data_train.append(data_i[train_start:train_end, :, :])
            data_valid.append(data_i[valid_start:valid_end, :, :])
            data_test.append(data_i[test_start:test_end, :, :])
            data_pretrain.append(data_i[pretrain_start:pretrain_end, :, :])

            label_train.append(label_i[train_start:train_end])
            label_valid.append(label_i[valid_start:valid_end])
            label_test.append(label_i[test_start:test_end])

        data_train = np.concatenate(data_train, axis=0)
        data_valid = np.concatenate(data_valid, axis=0)
        data_test = np.concatenate(data_test, axis=0)
        data_pretrain = np.concatenate(data_pretrain, axis=0)
        label_train = np.concatenate(label_train, axis=0)
        label_valid = np.concatenate(label_valid, axis=0)
        label_test = np.concatenate(label_test, axis=0)

    dataset_ = {}
    dataset_['data_pretrain'] = data_pretrain
    dataset_['data_train'] = data_train
    dataset_['data_valid'] = data_valid
    dataset_['data_test'] = data_test
    dataset_['label_train'] = label_train
    dataset_['label_valid'] = label_valid
    dataset_['label_test'] = label_test
    dataset_['n_class'] = n_class
    dataset_['n_dim'] = data_pretrain.shape[1]
    dataset_['data_len'] = data_pretrain.shape[2]
    return dataset_


def get_uea_data_names():
    data_names = [
        'ArticularyWordRecognition',
        'AtrialFibrillation',
        'BasicMotions',
        'CharacterTrajectories',
        'Cricket',
        'DuckDuckGeese',
        'EigenWorms',
        'Epilepsy',
        'ERing',
        'EthanolConcentration',
        'FaceDetection',
        'FingerMovements',
        'HandMovementDirection',
        'Handwriting',
        'Heartbeat',
        'InsectWingbeat',
        'JapaneseVowels',
        'Libras',
        'LSST',
        'MotorImagery',
        'NATOPS',
        'PEMS-SF',
        'PenDigits',
        'PhonemeSpectra',
        'RacketSports',
        'SelfRegulationSCP1',
        'SelfRegulationSCP2',
        'SpokenArabicDigits',
        'StandWalkJump',
        'UWaveGestureLibrary',
    ]
    return data_names

