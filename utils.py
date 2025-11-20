from itertools import combinations
from math import sqrt

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist
from scipy.stats import norm, rv_discrete, sobol_indices
from sklearn.utils.extmath import randomized_svd, squared_norm


def multi_replace(mat, delta=None):
    z_mat = mat == 0
    tot = np.sum(z_mat, axis=-1, keepdims=True)
    if delta is None:
        delta = np.min(mat[mat > 0]) * 0.65
    zcnts = 1 - tot * delta
    mat = np.where(z_mat, delta, zcnts * mat)
    return mat


def clr(mat, axis=-1):
    lmat = np.log(mat)
    gm = np.mean(lmat, axis=axis, keepdims=True)
    return lmat - gm


def clr_inv(mat, axis=-1):
    emat = np.exp(mat - np.max(mat, axis=axis, keepdims=True))
    emat = emat / np.sum(emat, axis=axis, keepdims=True)
    return emat


def nanmean(arr, axis=None):
    counts = np.sum(~np.isnan(arr), axis=axis)
    sums = np.nansum(arr, axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = sums / counts
    return result


def nansum(arr, axis=None):
    notallnan = (np.any(~np.isnan(arr), axis=axis)).astype(int)
    sums = np.nansum(arr, axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = sums / notallnan
    return result


def get_vars(obj):
    cls = obj.__class__
    instance_attrs = obj.__dict__
    class_attrs = {
        k: v
        for k, v in cls.__dict__.items()
        if not callable(v) and not k.startswith("__") and k not in instance_attrs
    }
    return class_attrs | instance_attrs


def pw_dist(X, metric):
    if len(X) == 1:
        return np.zeros(1)
    D = pdist(X, metric)
    if metric == "jensenshannon":
        D = np.square(D)
        D = D / np.log(2)
    return D


pw_cos_min = lambda X: pw_dist(X, "cosine").min().item()
pw_jsd_min = lambda X: pw_dist(X, "jensenshannon").min().item()
pw_bcd_min = lambda X: pw_dist(X, "braycurtis").min().item()
pw_cos_mean = lambda X: pw_dist(X, "cosine").mean().item()
pw_jsd_mean = lambda X: pw_dist(X, "jensenshannon").mean().item()
pw_bcd_mean = lambda X: pw_dist(X, "braycurtis").mean().item()


def pw_topic_distance(B_lst, metric="jensenshannon"):
    D_lst = []
    for i, j in combinations(range(len(B_lst)), 2):
        D = cdist(B_lst[i], B_lst[j], metric)
        D = np.nan_to_num(D)
        if metric == "jensenshannon":
            D = D / np.sqrt(np.log(2))
        row_ind, col_ind = linear_sum_assignment(D)
        D_lst.append(D[row_ind, col_ind].mean())
    return np.array(D_lst)

def nndsvd_init(X, n_components, random_state=None):
    norm = lambda x: sqrt(squared_norm(x))
    U, S, V = randomized_svd(X, n_components, random_state=random_state)
    W = np.zeros_like(U)
    H = np.zeros_like(V)
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])
    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)
        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n
        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v
    H = H / H.sum(axis=1, keepdims=True)
    H = multi_replace(H, 1e-10)
    B = np.log(H)
    B = B - np.mean(B, axis=1, keepdims=True)
    return B

def profile_log_likelihood(evals):
    kmax = len(evals)
    ll = np.zeros(kmax - 1)
    for k in range(kmax - 1):
        group1 = evals[0 : k + 1]
        group2 = evals[k + 1 : kmax]
        mu1 = np.mean(group1)
        mu2 = np.mean(group2)
        var = (np.sum((group1 - mu1) ** 2) + np.sum((group2 - mu2) ** 2)) / kmax
        std = np.sqrt(var)
        ll_group1 = np.sum(norm.logpdf(group1, mu1, std))
        ll_group2 = np.sum(norm.logpdf(group2, mu2, std))
        ll[k] = ll_group1 + ll_group2
    return ll

def get_sobol_indices(func, dataset, attrs):
    dist_time = rv_discrete(
        values=(
            np.squeeze(dataset.scaler.transform(attrs.timepoints[:, np.newaxis])),
            np.full(attrs.n_steps, 1 / attrs.n_steps),
        )
    )
    dist_cat = [
        rv_discrete(
            values=(
                np.arange(n_cat),
                np.full(n_cat, 1 / n_cat),
            )
        )
        for n_cat in attrs.n_cat_lst
    ]
    dists = [dist_time, *dist_cat]
    rng = np.random.default_rng(seed=0)
    sobol_result = sobol_indices(
        func=func,
        n=2**12,
        dists=dists,
        rng=rng,
    )
    var_Y = np.var(
        np.hstack([sobol_result._f_A, sobol_result._f_B]), axis=1, ddof=1, keepdims=True
    )
    sobol_agg = np.sum(sobol_result.total_order * var_Y / var_Y.sum(), axis=0)
    return sobol_result, sobol_agg
