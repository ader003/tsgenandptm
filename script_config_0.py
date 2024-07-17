# -*- coding: utf-8 -*-
"""
@author: <blinded>
"""


import os
import pathlib
from engine.config import get_dataset
from engine.config import get_dist_classifier
from engine.config import get_base_classifier
from engine.config import get_pretrain_model
from engine.config import get_generative


def main():
    uea_dir = 'PATH_TO_MULTIVARIATE_ARCHIVE_HERE'
    ucr_dir = "PATH_TO_UCR_ARCHIVE_HERE"

    config_dir = os.path.join('..', 'release_ptm_data_gen/config/')
    path = pathlib.Path(config_dir)
    path.mkdir(parents=True, exist_ok=True)

    get_dataset(config_dir, ucr_dir, uea_dir)
    get_dist_classifier(config_dir)
    get_base_classifier(config_dir)
    get_pretrain_model(config_dir)
    get_generative()


if __name__ == '__main__':
    main()

