# -*- coding: utf-8 -*-
"""
@File: train_brain.py
@Time: 2020-03-16 22:47
@Desc: train_brain.py
"""
try:
    import argparse
    import numpy as np
    import time

    from model import VIModel_PY
    from config import DATA_PARAMS, BRAIN_DIR
    from utils import console_log
    from datasets import get_adhd_data
    from cluster_process import ClusterProcess

except ImportError as e:
    print(e)
    raise ImportError


if __name__ == "__main__":

    # set args
    parser = argparse.ArgumentParser(prog='HPY-datas',
                                    description='Hierarchical Pitman-Yor process Mixture Models of datas Distributions')
    parser.add_argument('-c', '--algorithm_category', dest='algorithm_category', help='choose VIModel_PY:0',
                        default=0, type=int)
    parser.add_argument('-name', '--data_name', dest='data_name', help='data_name', default='adhd')
    parser.add_argument('-lp', '--load_params', dest='load_params', help='load_params', default=1, type=int)
    parser.add_argument('-verbose', '--verbose', dest='verbose', help='verbose', default=1, type=int)
    # hyper parameters
    parser.add_argument('-k', '--K', dest='K', help='truncation level K', default=12, type=int)
    parser.add_argument('-t', '--T', dest='T', help='truncation level T', default=5, type=int)
    parser.add_argument('-z', '--zeta', dest='zeta', help='zeta', default=0.01, type=float)
    parser.add_argument('-u', '--u', dest='u', help='u', default=0.9, type=float)
    parser.add_argument('-v', '--v', dest='v', help='v', default=0.01, type=float)
    parser.add_argument('-tau', '--tau', dest='tau', help='top stick tau', default=1, type=float)
    parser.add_argument('-gamma', '--gamma', dest='gamma', help='second stick gamma', default=1, type=float)
    parser.add_argument('-th', '--threshold', dest='threshold', help='second threshold', default=1e-7, type=float)
    parser.add_argument('-mth', '--mix_threshold', dest='mix_threshold', help='mix_threshold', default=0.01, type=float)
    parser.add_argument('-sm', '--second_max_iter', dest='second_max_iter',
                        help='second max iteration of variational inference', default=-1, type=int)
    parser.add_argument('-m', '--max_iter', dest='max_iter', help='max iteration of variational inference', default=100, type=int)
    args = parser.parse_args()

    (K, T, mix_threshold, algorithm_category, max_iter, second_max_iter, threshold, group, dim) \
        = DATA_PARAMS[args.data_name]

    print('begin training......')
    print('========================dataset is {}========================'.format(args.data_name))

    if int(args.load_params) == 1:
        args.K = K
        args.T = T
        args.mix_threshold = mix_threshold
        args.algorithm_category = algorithm_category
        args.second_max_iter = second_max_iter
        args.threshold = threshold
        args.max_iter = max_iter

    # py process
    # ================================================================================================================ #
    args.tau = 10
    args.gamma = 1
    args.omega = 0.2
    args.eta = 0.5
    args.u = 0.9
    args.v = 0.01
    args.zeta = 0.01
    func_filenames = get_adhd_data(data_dir=BRAIN_DIR, n_subjects=30)
    cp = ClusterProcess(model=VIModel_PY(args), n_components=30, smoothing_fwhm=12.,
                        memory="nilearn_cache", threshold=1., memory_level=2,
                        verbose=10, random_state=0)
    b = time.time()
    cp.fit(func_filenames)
    train_data = cp.train_data
    pred, container, pro = cp.model.predict_brain(train_data[0:1])
    e = time.time()
    print(e - b)
    # cp.plot_pro(pro.T, save=False, name='vmf-py', item_file='sub{}'.format(1))
    cp.plot_all(pred, save=True, name='vmf-py', item_file='sub{}'.format(1))
    ca = np.unique(pred)
    print(ca)
    measure_dict = console_log(pred=pred[:12000], data=train_data[0][:12000], model_name='HPY-VMF-brain')

