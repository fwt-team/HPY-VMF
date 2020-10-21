# -*- coding: utf-8 -*-
"""
@File: datasets.py
@Time: 2020-03-17 22:17
@Desc: datasets.py
"""
try:
    from nilearn import datasets
except ImportError as e:
    print(e)
    raise ImportError


def get_adhd_data(data_dir='./datas/brain', n_subjects=1):

    dataset = datasets.fetch_adhd(data_dir=data_dir, n_subjects=n_subjects)
    imgs = dataset.func

    return imgs
