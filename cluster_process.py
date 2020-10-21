# -*- coding: utf-8 -*-
"""
@File: cluster_process.py
@Time: 2020-03-14 15:45
@Desc: cluster_process.py
"""
try:
    import warnings
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    from nilearn.decomposition.multi_pca import MultiPCA
    from nilearn.plotting import show, plot_stat_map, plot_prob_atlas
    from nilearn.image import iter_img
    from nilearn._utils.compat import Memory

    from config import RESULT_DIR
except ImportError as e:
    print(e)
    raise ImportError


class ClusterProcess(MultiPCA):

    def __init__(self, model, mask=None, n_components=20, smoothing_fwhm=6,
                 do_cca=True,
                 threshold='auto',
                 n_init=10,
                 random_state=None,
                 standardize=True, detrend=True,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='epi', mask_args=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0):
        super(ClusterProcess, self).__init__(
            n_components=n_components,
            do_cca=do_cca,
            random_state=random_state,
            # feature_compression=feature_compression,
            mask=mask, smoothing_fwhm=smoothing_fwhm,
            standardize=standardize, detrend=detrend,
            low_pass=low_pass, high_pass=high_pass, t_r=t_r,
            target_affine=target_affine, target_shape=target_shape,
            mask_strategy=mask_strategy, mask_args=mask_args,
            memory=memory, memory_level=memory_level,
            n_jobs=n_jobs, verbose=verbose)

        self.model_ = model
        self.train_data = None
        self.model = None

    def hvmf_fit(self, data):

        self.train_data = data / np.linalg.norm(data, axis=2, keepdims=True)
        data = self.train_data[:, 1000:1500, :]

        self.model = self.model_.fit(data)

        return self

    def _raw_fit(self, data):

        data = data.reshape((30, self.n_components, -1))
        self.hvmf_fit(data.transpose((0, 2, 1)))
        return self

    def plot_pro(self, ita, save=False, item_file='group', name='vmf', choose=None, cut_coords=None, display_mode='ortho', belong='1'):

        re_path = '{}/brain/{}/{}'.format(RESULT_DIR, name, item_file)
        if not os.path.exists(re_path):
            os.makedirs(re_path)

        for component in ita:
            if component.max() < -component.min():
                component *= -1
        if hasattr(self, "masker_"):
            self.components_img_ = self.masker_.inverse_transform(ita)

        components_img = self.components_img_
        warnings.filterwarnings("ignore")
        display = plot_prob_atlas(components_img, title='All components', view_type='filled_contours')
        if save:
            display.savefig('{}/pro.png'.format(re_path), dpi=200)

        name = ['vmf-py', 'vmf-dp', 'gmm-dp']
        for i, cur_img in enumerate(iter_img(components_img)):
            if cut_coords is None:
                display = plot_stat_map(cur_img, dim=-.5, display_mode=display_mode, threshold=1e-2,
                                        cmap=plt.get_cmap('autumn'))
            else:
                display = plot_stat_map(cur_img, cut_coords=cut_coords, display_mode=display_mode, dim=-.5,
                                        threshold=1e-2,
                                        cmap=plt.get_cmap('autumn'))
            if save:
                if choose is not None and belong is not None:
                    display.savefig('{}/{}-{}-item{}.png'.format(re_path, name[i], belong, choose[i]), dpi=200)
                elif choose is not None:
                    display.savefig('{}/item{}.png'.format(re_path, choose[i] + 1), dpi=200)
                elif belong is not None:
                    display.savefig('{}/{}-item{}.png'.format(re_path, belong, i + 1), dpi=200)
                else:
                    display.savefig('{}/item{}.png'.format(re_path, i + 1), dpi=200)
        if save is False:
            show()

    def plot_all(self, pred, N=40, save=False, item_file='group', name='vmf'):

        data = np.zeros((N, pred.shape[0]))
        total = 0
        for i in range(N):
            data[i][pred != i] = 0
            data[i][pred == i] = 1
            total += data[i][data[i] != 0].shape[0]

        print(total)

        if hasattr(self, "masker_"):
            self.components_img_ = self.masker_.inverse_transform(data)

        components_img = self.components_img_
        warnings.filterwarnings("ignore")
        display = plot_prob_atlas(components_img, title='All components', view_type='filled_contours')
        if save:
            re_path = '{}/brain/{}/{}'.format(RESULT_DIR, name, item_file)
            if not os.path.exists(re_path):
                os.makedirs(re_path)
            display.savefig('{}/all.png'.format(re_path), dpi=200)
        else:
            show()

