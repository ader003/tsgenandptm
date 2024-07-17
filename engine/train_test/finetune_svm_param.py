import numpy as np
import time
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.svm import SVC, LinearSVC
from .util import _get_embeddings, _get_data, _get_checkpoint
import random
import os

def _train_tune(dataset, model, batch_size, c_type='linear'):
    data_train, label_train, data_valid, label_valid, _, _ = _get_data(dataset)

    embeddings_train = _get_embeddings(data_train, model, batch_size)
    embeddings_valid = _get_embeddings(data_valid, model, batch_size)

    all_embeddings = np.concatenate((embeddings_train,embeddings_valid))
    all_labels = np.concatenate((label_train,label_valid))
    cv = np.zeros(len(all_labels))
    cv[:len(label_train)] = -1

    grid = None
    if c_type == 'linear':
        param_grid = {'C': [2**-4,2**-3,2**-2,1,2,4,8,16]} # include default C
        grid = GridSearchCV(LinearSVC(), param_grid, refit = True, verbose = 3, cv=PredefinedSplit(cv))
    elif c_type == 'nonlinear':
        param_grid = {'C': [2**-4,2**-3,2**-2,1,2,4,8,16],
        'gamma': [2**-4,2**-3,2**-2,1,4,8,16], # 2^-4 to 2^-4 # include default C
        'kernel': ['rbf']} 
        grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3, cv=PredefinedSplit(cv))

    grid.fit(all_embeddings, all_labels)

    return grid.best_estimator_, grid.best_score_


def train_tune_svm_theta(dataset, model, model_path, train_config, result_path, c_type='linear'):
    n_epoch = int(train_config['n_epoch'])
    n_ckpt = int(train_config['n_ckpt'])
    batch_size = int(train_config['batch_size'])
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

    data_test = dataset['data_test']
    label_test = dataset['label_test']
    label_valid = dataset['label_valid']
    for i in ckpts:
        model_path_i = model_path.format(i)
        result_path_i = result_path.format(i)
        if not os.path.isfile(result_path_i):
            print("SVM Tuning on Model: {}".format(model_path_i))
            ckpt = torch.load(model_path_i)
            model.state_dict(ckpt['model_state_dict'])
            embeddings_test = _get_embeddings(data_test, model, batch_size)
            # select hyperparams and then test
            tic = time.time()
            best_estimator, acc_valid = _train_tune(dataset, model, batch_size, c_type=c_type)
            acc_test = best_estimator.score(embeddings_test, label_test)
            time_elapsed = time.time() - tic
            
            np.savez(result_path_i,
                        label_valid=label_valid,
                        label_test=label_test,
                        # predict_valid=predict_valid,
                        # predict_test=predict_test,
                        acc_valid=acc_valid,
                        acc_test=acc_test,
                        time_elapsed=time_elapsed)
            print((f'{result_path_i}, {acc_test:0.4f}, {time_elapsed:0.4f}'))

    return

