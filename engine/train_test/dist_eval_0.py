# -*- coding: utf-8 -*-
"""
@author:  <blinded>
"""


import os
import time
import numpy as np
from .util import _normalize_dataset
from sklearn.svm import SVC
# from tslearn.svm import TimeSeriesSVC
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sklearn.neighbors import KNeighborsClassifier

# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline

def _reshape_data(data):
    n_data = data.shape[0]
    n_dim = data.shape[1]
    data_len = data.shape[2]
    data = np.reshape(data, [n_data, n_dim * data_len, ])
    return data


def _get_predict(data, label, model):
    tic = time.time()
    predict = model.predict(data)
    predict_time = time.time() - tic
    acc = np.sum(predict == label) / label.shape[0]
    return predict, acc, predict_time


def dist_eval(dataset, method_config, result_path):
    data_train = dataset['data_train']
    data_valid = dataset['data_valid']
    data_test = dataset['data_test']

    label_train = dataset['label_train']
    label_valid = dataset['label_valid']
    label_test = dataset['label_test']

    if os.path.isfile(result_path):
        result = np.load(result_path, allow_pickle=True)
        acc_valid = result['acc_valid']
        acc_test = result['acc_test']
        time_valid = result['time_valid']
        time_test = result['time_test']
        print((f'{result_path}, {acc_valid:0.4f}, {acc_test:0.4f}, '
               f'{time_valid+time_test:0.2f}'))
        return

    data_train = _normalize_dataset(data_train)
    data_valid = _normalize_dataset(data_valid)
    data_test = _normalize_dataset(data_test)

    metric = method_config['model']['metric']
    c_type = "nn"
    assert metric in ['ed', 'dtw', ]

    model = None
    if metric == 'ed':
        data_train = _reshape_data(data_train)
        data_valid = _reshape_data(data_valid)
        data_test = _reshape_data(data_test)

        if c_type == "nn":
            model = KNeighborsClassifier(n_neighbors=1,n_jobs=-1)
    elif metric == 'dtw':
        if c_type == "nn":
            data_train = np.swapaxes(data_train, 1, 2)
            data_valid = np.swapaxes(data_valid, 1, 2)
            data_test = np.swapaxes(data_test, 1, 2)
            model = KNeighborsTimeSeriesClassifier(n_neighbors=1, n_jobs=-1)

    model.fit(data_train, label_train)
    predict_valid, acc_valid, time_valid = _get_predict(
        data_valid, label_valid, model)
    predict_test, acc_test, time_test = _get_predict(
        data_test, label_test, model)

    np.savez(result_path,
             label_valid=label_valid,
             label_test=label_test,
             predict_valid=predict_valid,
             predict_test=predict_test,
             acc_valid=acc_valid,
             acc_test=acc_test,
             time_valid=time_valid,
             time_test=time_test)
    print((f'{result_path}, {acc_valid:0.4f}, {acc_test:0.4f}, '
           f'{time_valid+time_test:0.2f}'))

