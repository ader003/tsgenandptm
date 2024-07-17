# -*- coding: utf-8 -*-
"""
@author: <blinded>
"""


import os
import numpy as np
import sktime.datasets
import sktime.datatypes


def print_data_state(output_path, data_name):
    npz = np.load(output_path)
    n_dim = npz['data_train'].shape[1]
    n_train = npz['data_train'].shape[0]
    n_test = npz['data_test'].shape[0]
    ts_len = npz['data_train'].shape[2]
    print(f"{data_name}, {n_dim}, {n_train}, {n_test}, {ts_len}")


def convert_to_numpy(data):
    n_data = data.shape[0]
    n_feature = data.shape[1]
    data_len = 0
    data_out = []
    for i in range(n_data):
        data_out.append([])
        for j in range(n_feature):
            data_ij = data.iloc[i].iloc[j].to_numpy()
            data_out[i].append(data_ij)
            data_len_ij = data_out[i][j].shape[0]
            if data_len_ij > data_len:
                print(f'{data_len}->{data_len_ij}')
                data_len = data_len_ij

    data_out_ = np.zeros((n_data, n_feature, data_len,))
    for i in range(n_data):
        for j in range(n_feature):
            data_len_ij = data_out[i][j].shape[0]
            data_out_[i, j, :data_len_ij] = data_out[i][j]
    return data_out_


def main():
    data_dir = '/path/to/Multivariate_ts'
    output_dir = '/path/to/output_dir'

    data_names = [
        'LSST',
        'SelfRegulationSCP2',
        'FaceDetection',
        'MotorImagery',
        'PenDigits',
        'Libras',
        'PhonemeSpectra',
        'Cricket',
        'Handwriting',
        'ArticularyWordRecognition',
        'StandWalkJump',
        'ERing',
        'HandMovementDirection',
        'SelfRegulationSCP1',
        'Heartbeat',
        'RacketSports',
        'EigenWorms',
        'FingerMovements',
        'PEMS-SF',
        'Epilepsy',
        'NATOPS',
        'AtrialFibrillation',
        'EthanolConcentration',
        'BasicMotions',
        'DuckDuckGeese',
        'UWaveGestureLibrary',
    ]

    for data_name in data_names:
        output_path = os.path.join(
            output_dir, data_name + '.npz')
        if os.path.isfile(output_path):
            print_data_state(output_path)
            continue

        ts_path = os.path.join(
            data_dir, data_name, data_name + '_TRAIN.ts')
        data_train, label_train = sktime.datasets.load_from_tsfile(
            ts_path, return_y=True, return_data_type="np3D")
        ts_path = os.path.join(
            data_dir, data_name, data_name + '_TEST.ts')
        data_test, label_test = sktime.datasets.load_from_tsfile(
            ts_path, return_y=True, return_data_type="np3D")

        np.savez(output_path, data_train=data_train, label_train=label_train,
                 data_test=data_test, label_test=label_test)
        print_data_state(output_path, data_name)

    data_names = [
        'InsectWingbeat',
        'CharacterTrajectories',
        'JapaneseVowels',
        'SpokenArabicDigits',
    ]

    for data_name in data_names:
        output_path = os.path.join(
            output_dir, data_name + '.npz')
        if os.path.isfile(output_path):
            print_data_state(output_path)
            continue

        ts_path = os.path.join(
            data_dir, data_name, data_name + '_TRAIN.ts')
        data_train, label_train = sktime.datasets.load_from_tsfile(
            ts_path, return_y=True)
        data_train = convert_to_numpy(data_train)
        ts_path = os.path.join(
            data_dir, data_name, data_name + '_TEST.ts')
        data_test, label_test = sktime.datasets.load_from_tsfile(
            ts_path, return_y=True)
        data_test = convert_to_numpy(data_test)

        np.savez(output_path, data_train=data_train, label_train=label_train,
                 data_test=data_test, label_test=label_test)
        print_data_state(output_path)


if __name__ == '__main__':
    main()
