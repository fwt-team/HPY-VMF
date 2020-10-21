# -*- coding: utf-8 -*-
# @file: model.py
# @time: 2020/1/13 15:16
# @desc: model.py
try:
    import numpy as np

    from scipy.special import digamma, iv, gammaln
    izip = zip

    from utils import predict, d_besseli, d_besseli_low, log_normalize, predict_pro
except ImportError as e:
    print(e)
    raise ImportError


class VIModel_PY:
    """
    Variational Inference Pitman-Yor process Mixture Models of datas Distributions
    """
    def __init__(self, args):

        self.K = args.K
        self.T = args.T
        self.newJ = self.K
        self.max_k = 700
        self.second_max_iter = args.second_max_iter
        self.args = args
        self.J = 3
        self.N = 300
        self.D = 3
        self.prior = dict()

        self.u = None
        self.v = None
        self.zeta = None
        self.xi = None
        self.k = None
        self.pi = None

        self.rho = None
        self.var_theta = None
        self.a = None
        self.b = None
        self.g = None
        self.h = None

        self.temp_top_stick = None
        self.temp_xi_ss = None
        self.temp_k_ss = None

        self.container = {
            'rho': [],
            'var_theta': []
        }

    def init_top_params(self, data):

        self.J = len(data)
        self.D = data[0].shape[1]

        while np.isfinite(iv(self.D / 2, self.max_k + 10)):
            self.max_k = self.max_k + 10

        total_data = np.vstack((i for i in data))
        self.prior = {
            'mu': np.sum(total_data, 0) / np.linalg.norm(np.sum(total_data, 0)),
            'zeta': self.args.zeta,
            'u': self.args.u,
            'v': self.args.v,
            'tau': self.args.tau,
            'gamma': self.args.gamma,
            'omega': self.args.omega,
            'eta': self.args.eta,
        }

        self.u = np.ones(self.K) * self.prior['u']
        self.v = np.ones(self.K) * self.prior['v']
        self.zeta = np.ones(self.K)
        self.xi = np.ones((self.K, self.D))

        self.xi = self.xi / np.linalg.norm(self.xi, axis=1)[:, np.newaxis]
        self.k = self.u / self.v

        self.a = np.ones(self.K)
        self.b = np.ones(self.K)
        self.temp_top_stick = np.zeros(self.K)
        self.temp_xi_ss = np.zeros((self.K, self.D))
        self.temp_k_ss = np.zeros(self.K)

        self.init_update(data)

    def set_temp_zero(self):

        self.temp_top_stick.fill(0.0)
        self.temp_xi_ss.fill(0.0)
        self.temp_k_ss.fill(0.0)

    def init_update(self, x):

        self.var_theta = np.ones((self.T, self.K)) * (1 / self.K)

        for i in range(self.J):
            N = x[i].shape[0]
            self.rho = np.ones((N, self.T)) * (1 / self.T)
            self.temp_top_stick += np.sum(self.var_theta, 0)
            self.temp_k_ss += np.sum(self.rho.dot(self.var_theta), 0)
            self.temp_xi_ss += self.var_theta.T.dot(self.rho.T.dot(x[i]))

        self.update_zeta_xi()
        self.update_u_v()
        self.update_a_b()

    def calculate_new_com(self):

        threshold = self.args.mix_threshold

        index = np.where(self.pi > threshold)[0]
        self.pi = self.pi[self.pi > threshold]
        self.newJ = self.pi.size

        self.xi = self.xi[index]
        self.k = self.k[index]

        if self.args.verbose:
            print("new component is {}".format(self.newJ))

    def init_second_params(self, N):

        self.rho = np.ones((N, self.T)) * (1 / self.T)

        self.g = np.zeros(self.T)
        self.h = np.zeros(self.T)

        self.update_g_h(self.rho)

    def expect_log_sticks(self, a, b, k):

        E_log_1_pi = np.roll(np.cumsum(digamma(b) - digamma(a + b)), 1)
        E_log_1_pi[0] = 0
        return digamma(a) - digamma(a + b) + E_log_1_pi

    def var_inf_2d(self, x, Elogsticks_1nd, ite):

        D = self.D
        Elog_phi = ((x.dot((self.xi * (self.u / self.v)[:, np.newaxis]).T)) +
                    (D / 2 - 1) * (digamma(self.u) - np.log(self.v)) -
                    (D / 2 * np.log(2 * np.pi)) -
                    (d_besseli(D / 2 - 1, self.k)) * (self.u / self.v - self.k) -
                    np.log(iv((D / 2 - 1), self.k) + np.exp(-700)))

        second_max_iter = 5000 if self.second_max_iter == -1 else self.second_max_iter
        self.init_second_params(x.shape[0])
        likelihood = 0.0
        old_likelihood = 1
        converge = 1
        Elogsticks_2nd = self.expect_log_sticks(self.g, self.h, self.T)
        for i in range(second_max_iter):
            # compute var_theta

            self.var_theta = self.rho.T.dot(Elog_phi) + Elogsticks_1nd
            log_var_theta, log_n = log_normalize(self.var_theta)
            self.var_theta = np.exp(log_var_theta)

            self.rho = self.var_theta.dot(Elog_phi.T).T + Elogsticks_2nd
            log_rho, log_n = log_normalize(self.rho)
            self.rho = np.exp(log_rho)

            self.update_g_h(self.rho)
            Elogsticks_2nd = self.expect_log_sticks(self.g, self.h, self.T)

            likelihood = 0.0
            # compute likelihood
            likelihood += np.sum((Elogsticks_1nd - log_var_theta) * self.var_theta)

            v = np.vstack((self.g, self.h))
            log_alpha = np.log(self.prior['gamma'])
            likelihood += (self.T - 1) * log_alpha
            dig_sum = digamma(np.sum(v, 0))
            likelihood += np.sum((np.array([1.0, self.prior['gamma']])[:, np.newaxis] - v) * (digamma(v) - dig_sum))
            likelihood -= np.sum(gammaln(np.sum(v, 0))) - np.sum(gammaln(v))

            # Z part
            likelihood += np.sum((Elogsticks_2nd - log_rho) * self.rho)

            # X part, the data part
            likelihood += np.sum(self.rho.T * np.dot(self.var_theta, Elog_phi.T))

            if i > 0:
                converge = (likelihood - old_likelihood) / abs(old_likelihood)
            old_likelihood = likelihood

            if converge < self.args.threshold:
                break

        self.temp_top_stick += np.sum(self.var_theta, 0)
        self.temp_k_ss += np.sum(self.rho.dot(self.var_theta), 0)
        self.temp_xi_ss += self.var_theta.T.dot(self.rho.T.dot(x))

        if ite == self.args.max_iter - 1:
            self.container['rho'].append(self.rho)
            self.container['var_theta'].append(self.var_theta)

        return likelihood

    def var_inf(self, x):

        for ite in range(self.args.max_iter):

            self.set_temp_zero()
            Elogsticks_1nd = self.expect_log_sticks(self.a, self.b, self.K)
            for i in range(self.J):
                self.var_inf_2d(x[i], Elogsticks_1nd, ite)

            self.optimal_ordering()
            # compute k
            self.k = self.u / self.v
            self.k[self.k > self.max_k] = self.max_k

            self.update_zeta_xi()
            self.update_u_v()
            self.update_a_b()

            if self.args.verbose == 1:
                print('=====> iteration: {}'.format(ite))
            if ite == self.args.max_iter - 1:
                # compute k
                # self.k = self.u / self.v
                # self.k[self.k > self.max_k] = self.max_k
                self.pi = np.exp(self.expect_log_sticks(self.a, self.b, self.K))
                self.calculate_new_com()
                if self.args.verbose:
                    print('mu: {}'.format(self.xi))
                    print('kappa: {}'.format(self.k))

    def optimal_ordering(self):

        s = [(a, b) for (a, b) in izip(self.temp_top_stick, range(self.K))]
        x = sorted(s, key=lambda y: y[0], reverse=True)
        idx = [y[1] for y in x]
        self.temp_top_stick[:] = self.temp_top_stick[idx]
        self.temp_k_ss[:] = self.temp_k_ss[idx]
        self.temp_xi_ss[:] = self.temp_xi_ss[idx]

    def update_u_v(self):

        D = self.D
        # compute u, v
        self.u = self.prior['u'] + (D / 2 - 1) * self.temp_k_ss + \
                 self.zeta * self.k * (d_besseli_low(D / 2 - 1, self.zeta * self.k))
        self.v = self.prior['v'] + self.temp_k_ss * (d_besseli(D / 2 - 1, self.k)) + \
                 self.prior['zeta'] * (d_besseli(D / 2 - 1, self.prior['zeta'] * self.k))

    def update_zeta_xi(self):

        # compute zeta, xi
        temp = np.expand_dims(self.prior['zeta'] * self.prior['mu'], 0) + self.temp_xi_ss
        self.zeta = np.linalg.norm(temp, axis=1)
        self.xi = temp / self.zeta[:, np.newaxis]

    def update_g_h(self, rho):

        N_k = np.sum(rho, 0)
        self.g = 1 + N_k - self.prior['eta']
        for i in range(self.T):
            if i == self.T - 1:
                self.h[i] = self.prior['gamma'] + self.T * self.prior['eta']
            else:
                temp = rho[:, i + 1:self.T]
                self.h[i] = self.prior['gamma'] + np.sum(np.sum(temp, 1), 0) + (i+1) * self.prior['eta']

    def update_a_b(self):

        self.a = 1 + self.temp_top_stick - self.prior['omega']
        for i in range(self.K):
            if i == self.K - 1:
                self.b[i] = self.prior['tau'] + self.K * self.prior['omega']
            else:
                temp = self.temp_top_stick[i + 1:self.K]
                self.b[i] = self.prior['tau'] + np.sum(temp) + (i + 1) * self.prior['omega']

    def fit(self, data):

        self.init_top_params(data)
        self.var_inf(data)
        return self

    def predict(self, data):
        # predict
        data = np.vstack((i for i in data))
        pred = predict(data, mu=self.xi, k=self.k, pi=self.pi, n_cluster=self.newJ)
        return pred

    def predict_brain(self, data):
        # predict
        data = np.vstack((i for i in data))
        pro = predict_pro(data, mu=self.xi, k=self.k, pi=self.pi, n_cluster=self.newJ)
        pred = np.argmax(pro, axis=1)
        return pred, self.container, pro

    def fit_predict(self, data):
        self.fit(data)
        return self.predict(data)
