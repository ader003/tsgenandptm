# -*- coding: utf-8 -*-
"""
@author: <blinded>
"""

def get_r1d_setting():
    encoder = {}
    encoder['in_dim'] = 'None'
    encoder['out_dim'] = 128
    encoder['n_dim'] = 64
    encoder['block_type'] = 'alternative'
    encoder['norm'] = 'None'
    encoder['is_projector'] = 'False'
    encoder['project_norm'] = 'None'
    return encoder


def get_trf_setting_0():
    encoder = {}
    encoder['in_dim'] = 'None'
    encoder['out_dim'] = 128
    encoder['n_layer'] = 4
    encoder['n_dim'] = 64
    encoder['n_head'] = 8
    encoder['norm_first'] = 'True'
    encoder['is_pos'] = 'True'
    encoder['is_projector'] = 'False'
    encoder['project_norm'] = 'None'
    encoder['dropout'] = 0.0
    return encoder


def get_timefreq_setting():
    timefreq = {}
    timefreq['jitter_strength'] = 0.1
    timefreq['freq_ratio'] = 0.1
    timefreq['freq_strength'] = 0.1
    timefreq['project_norm'] = 'None'
    return timefreq


def get_ts2vec_setting():
    return {'ph': 0}


def get_mixup_setting():
    return {'ph': 0}


def get_simclr_setting():
    return {'ph': 0}


def get_timeclr_setting():
    timeclr = {}
    timeclr['aug_bank_ver'] = 0
    return timeclr


def get_classifier_setting(classifier_type='nn'):
    classifier = {}
    if classifier_type == 'nn':
        classifier['n_dim'] = 64
        classifier['n_layer'] = 2
    
    return classifier


def get_train_setting():
    train = {}
    train['lr'] = 0.0001
    train['batch_size'] = 64
    train['n_epoch'] = 400
    train['n_ckpt'] = 400
    return train

