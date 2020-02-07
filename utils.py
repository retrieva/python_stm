# This code is available under the MIT License.
# (c)2018-2019 Hiroki Iida / Retrieva Inc.


import numpy as np
import scipy as sp


def update_Hessian(K, q_z, c_dv, wd, theta, inv_Sigma):
    hessian = np.diag(-1.0 * np.dot(q_z[0:K-1, :], c_dv))
    hessian += np.dot(np.sqrt(c_dv) * q_z[0:K-1, :], (np.sqrt(c_dv) * q_z[0:K-1, :]).T)
    hessian += wd * np.diag(theta[0:K-1])
    hessian -= wd * np.outer(theta[0:K-1], theta[0:K-1]) + inv_Sigma
    return hessian


def eta_optim_obj(ndoc, K, x, phi, Sigma, mu, c_dv, wd):
    """
    ndoc: int
    x:K numpy array
    phi: KxV numpy array
    """
    diff = x[:K-1] - mu[ndoc, :K-1]
    x -= x.max()
    obj_fn = 0.5 * np.dot(diff.T, np.dot(np.linalg.inv(Sigma), diff))
    obj_fn -= np.dot(c_dv,  np.log(np.dot(np.exp(x), phi)))
    obj_fn += wd[ndoc] * np.log(np.sum(np.exp(x)))
    return obj_fn


def eta_optim_grad(ndoc, K, x, phi, Sigma, mu, c_dv, wd):
    """
    ndoc: int
    x:K numpy array
    phi: KxV numpy arrray
    """
    diff = x[:K-1] - mu[ndoc, :K-1]
    x -= x.max()
    q_z = np.exp(x)[:, np.newaxis] * phi
    q_z /= np.sum(q_z, axis=0)
    theta = np.exp(x) / np.sum(np.exp(x))
    grad_fn = -1.0 * np.dot(q_z, c_dv) + wd[ndoc] * theta
    grad_fn += np.append(np.dot(Sigma, diff), 0.0)
    return grad_fn


def update_eta(m, K, eta, phi, Sigma, mu, c_dv, wd):
    eta_sol_options = {"maxiter": 500, "gtol": 1e-6}
    obj = lambda x: eta_optim_obj(m, K, x, phi, Sigma, mu, c_dv, wd)
    grad = lambda x: eta_optim_grad(m, K, x, phi, Sigma, mu, c_dv, wd)
    result = sp.optimize.minimize(fun=obj, x0=eta, method='BFGS', jac=grad, options=eta_sol_options)
    eta = result.x
    eta_max = eta.max()
    eta -= eta_max
    theta = np.exp(eta)/np.sum(np.exp(eta))
    q_z = np.exp(eta)[:, np.newaxis] * phi
    q_z /= np.sum(q_z, axis=0)
    eta += eta.max()
    return (eta, theta, q_z)


def RVM_regression(Y, X, K, it_num=100):
    """
    Parameters
    ---------
    Y: NxK matrix of target value

    X: NxD matrix of data

    K: topic number

    it_num: repeat count

    sup: N is data number(so it is equivalent to document number)
         D is data-dimension

    Returns:
    --------
    W: updated weight of linear regression

    ref: VB INFERENCE FOR LINEAR/LOGISTIC REGRESSION JAN DRUGOWITSCH et al
    """
    # inv-gamma prior from thesis
    N = X.shape[0]
    D = X.shape[1]

    a0 = np.full(K, 0.01)
    b0 = np.full(K, 0.0001)
    c0 = np.full(K, 0.01)
    d0 = np.full((K, D), 0.001)

    a_N = a0 + 0.5 * N
    b_N = b0
    c_N = c0 + 0.5
    d_N = d0
    updater_inv_V_N = np.dot(X.T, X)

    W = np.zeros((D, K))

    updater_W = np.dot(X.T, Y)

    updater_b_N = np.sum(Y*Y, axis=0)

    for _ in range(it_num):
        inv_V_N = np.zeros((K, D, D))
        for k in range(K):
            inv_V_N[k, :, :] += np.diag(np.ones(D) * c_N[k] / d_N[k, :]) + updater_inv_V_N

        for k in range(K):
            W[:, k] = np.dot(np.linalg.inv(inv_V_N[k]), updater_W[:, k])

        for k in range(K):
            b_N[k] = b0[k] + 0.5 * (updater_b_N[k] - np.dot(W[:, k].T, np.dot(inv_V_N[k], W[:, k])))

        for k in range(K):
            d_N[k] = d0[k] + 0.5 * W[:, k] * W[:, k] * a_N[k] / b_N[k]

    return W
