# -*- coding: utf-8 -*-
"""
@author: Audrey Der
"""

import os

def get_datafeeder(config_dict, feeder_type, batch_size):
    config_dict['datafeeder'] = {}
    config_dict['datafeeder']['feeder_type'] = feeder_type
    config_dict['datafeeder']['batch_size'] = batch_size
    return config_dict
