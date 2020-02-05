 #!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c)2018 Hiroki Iida / Retrieva Inc.

import numpy as np
import scipy as sp
from scipy.special import logsumexp
import lda
import utils
from abc import ABCMeta, abstractmethod


class STM_base(metaclass=ABCMeta):
    def __init__(self, K, X, Y, docs, V, sigma, interact=True):
        self.X = X  # DxPx matrix (Px is the num of tags)
        self.K = K
        self.D = len(docs)
        if X is not None:
            P = X.shape[1]
            self.Gamma = np.zeros((P, K-1))  # parameter of topics prior

        if Y is None:
            Y = np.zeros(self.D, dtype=int)

        self.Y = Y  # Dx1 matrix

        self.mu = np.zeros((self.D, K))
        self.Sigma = np.diag(np.ones(K-1)) * sigma  # if zero, no update. so using diag.
        self.c_dv = np.zeros((self.D, V), dtype=int)
        self.wd = np.zeros(self.D, dtype=int)
        self.mv = np.zeros(V, dtype=int)
        for m, doc in enumerate(docs):
            for t in doc:
                self.c_dv[m, t] += 1
                self.wd[m] += 1
                self.mv[t] += 1

        self.mv = np.log(self.mv) - np.log(np.sum(self.mv))
        self.docs = docs
        self.docs_vocab = []
        for doc in docs:
            self.docs_vocab.append(sorted(list(set(doc))))

        self.V = V
        self.eta = np.zeros((self.D, K))
        self.theta = np.exp(self.eta) / np.sum(np.exp(self.eta), axis=1)[:, np.newaxis]

        self.A = np.unique(np.array(Y))
        self.len_A = len(self.A)
        self.phi = np.zeros((self.len_A, self.K, self.V))

    def lda_initialize(self, alpha, beta, itr, voca, smartinit=True):
        lda_init = lda.LDA(self.K, alpha, beta, self.docs, self.V, smartinit)
        lda_init.learning(itr, voca)

        Kalpha = self.K * alpha
        self.theta = lda_init.n_m_z / (np.vectorize(len)(np.array(self.docs)) + Kalpha)[:, np.newaxis]

        self.phi += lda_init.worddist()

        del lda_init

    def output_word_topic_dist(self, voca):
        def output(phi, voca):
            for k in range(self.K):
                print("\n-- topic: {}".format(k))
                for w in np.argsort(-phi[k])[:20]:
                    print("{}: {}".format(voca[w], phi[k, w]))

        phi = np.average(self.phi, axis=0)
        output(phi, voca)

    def perplexity(self, docs=None, Y=None):
        if docs is None:
            docs = self.docs
        if Y is None:
            Y = self.Y
        log_per = 0
        N = 0

        for m, (doc, a) in enumerate(zip(docs, Y)):
            for w in doc:
                log_per -= np.log(np.dot(self.phi[a, :, w], self.theta[m]))
            N += len(doc)
        return np.exp(log_per / N)

    def learning(self, iteration, voca):
        pre_perp = self.perplexity()
        print("initial perplexity=%f" % pre_perp)
        for i in range(iteration):
            self.inference(i)
            perp = self.perplexity()
            print("-%d p=%f" % (i + 1, perp))
            if pre_perp:
                if pre_perp < perp:
                    self.output_word_topic_dist(voca)
                    pre_perp = None
                else:
                    pre_perp = perp
        self.output_word_topic_dist(voca)

    def inference(self, iter_num):
        """learning once iteration"""
        # E-step
        # update q_eta and q_z
        phi_updater, q_v, variance_topics = self.update_Estep()
        # M-step
        self.update_mu_and_Gamma()

        # update Sigma
        if iter_num > 10:
            self.update_Sigma(q_v, variance_topics)

        # update phi
        self.update_phi(phi_updater)

    def update_Estep(self):
        E_count = np.zeros((len(self.A), self.K, self.V))
        q_v = np.zeros((self.K - 1, self.K - 1))
        variance_topics = np.zeros((self.K - 1, self.K - 1))
        inv_Sigma = np.linalg.inv(self.Sigma)

        for m, (_, i, a) in enumerate(zip(self.docs, self.docs_vocab, self.Y)):
            # because fuzzy index induces copy
            phi_a = self.phi[a, :, i].T
            c_dv_d = self.c_dv[m, i]
            self.eta[m], self.theta[m], q_z_d \
                = utils.update_eta(m, self.K, self.eta[m],
                                   phi_a, self.Sigma,
                                   self.mu, c_dv_d, self.wd)

            # prepare update Sigma(calc q_v) and phi(calc phi_tmp)
            E_count[a, :, i] += (c_dv_d * q_z_d).T
            hessian = utils.update_Hessian(self.K, q_z_d, c_dv_d, self.wd[m], self.theta[m], inv_Sigma)
            q_v += np.linalg.inv(hessian)
            diff_var_and_mean = self.calc_diff_var_and_mean(m)
            variance_topics += np.outer(diff_var_and_mean, diff_var_and_mean)
        return (E_count, q_v, variance_topics)

    @abstractmethod
    def update_mu_and_Gamma(self):
        pass

    def update_Sigma(self, q_v, variance_topics):
        self.Sigma = (q_v + variance_topics) / len(self.docs)

    @abstractmethod
    def update_phi(self, E_count):
        pass


class STM_jeff_base(STM_base):
    def __init__(self, K, X, Y, docs, V, sigma, interact=True):
        super().__init__(K, X, Y, docs, V, sigma, interact)

        self.aspectmod = self.len_A > 1.0
        self.interact = interact
        self.coef_row = self.K + self.len_A * self.aspectmod + self.len_A * self.K * self.interact

        self.kappa_params = np.zeros((self.coef_row, V))
        self.kappa_sum = np.full((self.len_A, self.K, self.V), self.mv)

    def jeffereysKappa(self, E_count):
        def kappa_obj(kappa_param, kappa_other, c_k, bigC_k, gaussprec):
            p1 = -1 * np.sum(c_k * kappa_param)
            demon_kappas = kappa_other * np.exp(kappa_param)
            lseout = np.log(np.sum(demon_kappas, axis=1))
            p2 = np.sum(bigC_k * lseout)
            p3 = 0.5 * np.sum(kappa_param**2 * gaussprec)
            return p1 + p2 + p3

        def kappa_grad(kappa_param, kappa_other, c_k, bigC_k, gaussprec):
            denom_kappas = kappa_other * np.exp(kappa_param)
            betaout = denom_kappas / np.sum(denom_kappas, axis=1)[:, np.newaxis]
            p2 = np.sum(bigC_k[:, np.newaxis] * betaout, axis=0)  # sum up the non focus axis
            p3 = kappa_param * gaussprec
            return -c_k + p2 + p3

        if(not(self.aspectmod)):
            KbyV = E_count[0]
            KbyA = np.sum(KbyV, axis=1)
        else:
            KbyV = np.sum(E_count, axis=0)
            KbyA = np.sum(E_count, axis=2).T

        max_it = 3
        tol = .001
        kappamax_it = 1000
        taumax_it = 1000
        tautol = 1e-5

        # define update indicater upmost
        i_update_kv = self.K
        if (self.aspectmod and self.interact):
            i_update_ka = self.K + self.len_A
            i_update_kav = self.coef_row
        else:
            i_update_ka = self.coef_row
            i_update_kav = 0

        opt_tau = np.vectorize(lambda x: 1/x**2 if x**2 > 1e-5 else 1e5)

        for it in range(max_it):
            compare = np.abs(self.kappa_params) < .001
            for i in range(self.coef_row):  # i:0~K-1=>update kv, K~K+A-1=>update ka, K+A~K+A+K*A-1=>update kav
                kappa_init = self.kappa_params[i]
                if i < i_update_kv:
                    k = i
                    c_k = KbyV[k, :]
                    bigC_k = KbyA[k, :]
                    self.kappa_sum[:, k, :] -= kappa_init
                    kappa_other = np.exp(self.kappa_sum[:, k, :])
                elif i < i_update_ka:
                    a = i - self.K
                    c_k = np.sum(E_count[a], axis=0)
                    bigC_k = KbyA[:, a]
                    self.kappa_sum[a, :, :] -= kappa_init
                    kappa_other = np.exp(self.kappa_sum[a, :, :])
                elif i < i_update_kav:
                    a, k = divmod(i-self.K-self.len_A, self.K)
                    c_k = E_count[a, k, :]
                    bigC_k = KbyA[k, a][np.newaxis]
                    self.kappa_sum[a, k, :] -= kappa_init
                    kappa_other = np.exp(self.kappa_sum[a, k, :])[np.newaxis, :]

                converged = False
                for j in range(taumax_it):
                    if(not(np.any(kappa_init))):
                        gaussprec = 1
                    else:
                        gaussprec = opt_tau(kappa_init)

                    result = sp.optimize.minimize(fun=kappa_obj, x0=kappa_init,
                                                  args=(kappa_other, c_k, bigC_k, gaussprec),
                                                  jac=kappa_grad, method="L-BFGS-B", options={'maxiter': kappamax_it})
                    kappa_init = result.x
                    converged = np.mean(np.abs(self.kappa_params[i] - kappa_init))
                    self.kappa_params[i] = kappa_init
                    if converged <= tautol:
                        break

                if i < i_update_kv:
                    self.kappa_sum[:, k, :] += self.kappa_params[i]
                elif i < i_update_ka:
                    self.kappa_sum[a, :, :] += self.kappa_params[i]
                elif i < i_update_kav:
                    self.kappa_sum[a, k, :] += self.kappa_params[i]

            current = np.abs(self.kappa_params) < .001
            sparseagree = np.average(compare == current)
            self.phi = np.exp(self.kappa_sum - logsumexp(self.kappa_sum, axis=2)[:, :, np.newaxis])
            if sparseagree > tol:
                break

    def update_phi(self, E_count):
        self.jeffereysKappa(E_count)

    @abstractmethod
    def calc_diff_var_and_mean(self, m):
        pass


class STM_jeff_reg(STM_jeff_base):
    def __init__(self, K, X, Y, docs, V, sigma, interact=True):
        super().__init__(K, X, Y, docs, V, sigma, interact)

    def calc_diff_var_and_mean(self, m):
        return (self.eta[m, 0:self.K-1] - np.dot(self.X, self.Gamma)[m])

    def update_mu_and_Gamma(self):
        tmp_Gamma = utils.RVM_regression(self.eta, self.X, self.K)
        self.Gamma = tmp_Gamma[:self.D, :self.K-1]
        self.mu = np.dot(self.X, self.Gamma)


class STM_jeff_noX(STM_jeff_base):
    def __init__(self, K, X, Y, docs, V, sigma, interact=True):
        super().__init__(K, X, Y, docs, V, sigma, interact)

    def calc_diff_var_and_mean(self, m):
        return (self.eta[m, 0:self.K-1] - self.mu[m, 0:self.K-1])

    def update_mu_and_Gamma(self):
        self.mu = np.tile(np.sum(self.eta, axis=0) / self.D, (self.D, 1))


class STM_noY_base(STM_base):
    def __init__(self, K, X, Y, docs, V, sigma, interact=True):
        super().__init__(K, X, Y, docs, V, sigma, interact)

    def calc_diff_var_and_mean(self, m):
        pass

    def update_phi(self, q_z):
        # ref: Variational EM Algorithms for Correlated Topic Models / Mohhammad Emtiaz Khan et al
        for k in range(self.K):
            self.phi[0, k, :] = q_z[0, k, :]

        self.phi[0, :, :] = q_z[0] / np.sum(q_z[0, :, :], axis=1)[:, np.newaxis]


class STM_noY_reg(STM_noY_base):
    def __init__(self, K, X, Y, docs, V, sigma, interact=True):
        super().__init__(K, X, Y, docs, V, sigma, interact)

    def calc_diff_var_and_mean(self, m):
        return (self.eta[m, 0:self.K-1] - np.dot(self.X, self.Gamma)[m])

    def update_mu_and_Gamma(self):
        tmp_Gamma = utils.RVM_regression(self.eta, self.X, self.K)
        self.Gamma = tmp_Gamma[:self.D, :self.K-1]
        self.mu = np.dot(self.X, self.Gamma)


class STM_noY_noX(STM_noY_base):
    def __init__(self, K, X, Y, docs, V, sigma, interact=True):
        super().__init__(K, X, Y, docs, V, sigma, interact)

    def calc_diff_var_and_mean(self, m):
        return (self.eta[m, 0:self.K-1] - self.mu[m, 0:self.K-1])

    def update_mu_and_Gamma(self):
        self.mu = np.tile(np.sum(self.eta, axis=0) / self.D, (self.D, 1))


def STM_factory_method(K, X, Y, docs, V, sigma, interact=True):
    if Y is None:
        if X is None:
            return STM_noY_noX(K, X, Y, docs, V, sigma, interact)
        else:
            return STM_noY_reg(K, X, Y, docs, V, sigma, interact)
    else:
        if X is None:
            return STM_jeff_noX(K, X, Y, docs, V, sigma, interact)
        else:
            return STM_jeff_reg(K, X, Y, docs, V, sigma, interact)
