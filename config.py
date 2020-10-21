# -*- coding: utf-8 -*-
"""
@File: config.py
@Time: 2020-02-15 00:05
@Desc: config.py
"""

import os

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

# Local directory for datasets
DATASETS_DIR = os.path.join(REPO_DIR, 'datas')

SYNC_DIR = os.path.join(DATASETS_DIR, 'synthetic')
BRAIN_DIR = os.path.join(DATASETS_DIR, 'brain')

RESULT_DIR = os.path.join(REPO_DIR, 'result')

# difference datasets config
# K, T, mix_threshold, algorithm_category, max_iter, second_max_iter, threshold, group, dim

DATA_PARAMS = {
    # For the evaluation of simulation data parameters, 500 rounds of iteration can ensure that the
    # parameters kappa evaluation is correct, but 10 rounds of iteration can achieve an accuracy of 100%.
    'small_data': (10, 5, 0.011, 0, 10, -1, 1e-7, 2, 3),
    'adhd': (150, 45, 0.01, 0, 13, -1, 1e-7, 30, 30),
}