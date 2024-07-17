# -*- coding: utf-8 -*-
"""
@author:  <blinded>
"""

def get_timefreq_setting(timefreq):
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


def get_timeclr_setting(timeclr):
    timeclr['aug_bank_ver'] = 0
    return timeclr


def get_classifier_setting(): # this needs to be handled differently than release_v2
    classifier = {}
    classifier['metric'] = 'dtw'
    # https://tslearn.readthedocs.io/en/latest/gen_modules/metrics/tslearn.metrics.cdist_gak.html#tslearn.metrics.cdist_gak
    classifier['c_type'] = 'nonlinear' 
    return classifier


def get_ptm(config_dict, PTM): # unsure of where config_path comes into play
    if PTM == None:
        config_dict["none"] = {}
        return config_dict
    elif PTM == "ts2vec":
        config_dict[PTM] = get_ts2vec_setting()
    elif PTM == "mixup":
        config_dict[PTM] = get_mixup_setting()
    elif PTM == "simclr":
        config_dict[PTM] = get_simclr_setting() # retired; not run in experimentation
    elif PTM == "timefreq":
        config_dict[PTM] = get_timefreq_setting({})
    elif PTM == "timeclr":
        config_dict[PTM] = get_timeclr_setting({})
    return config_dict
