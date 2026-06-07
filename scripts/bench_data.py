from types import SimpleNamespace

import numpy as np

AXES = {
    "samples": [1000, 2000, 4000, 8000, 16000],
    "features": [128, 256, 512, 1024, 2048],
    "subjects": [50, 100, 200, 400, 800],
    "components": [2, 4, 8, 16, 32],
}
DEFAULTS = {"samples": 2000, "features": 512, "subjects": 200, "components": 8}
N_CATEGORIES = 3


def resolve(axis, value):
    dims = dict(DEFAULTS)
    dims[axis] = value
    return dims["samples"], dims["features"], dims["subjects"], dims["components"]


def make_attrs(n_samples, n_features, n_subjects, n_components, seed=42):
    rng = np.random.default_rng(seed)

    x_num = rng.uniform(0.0, 1.0, (n_samples, 1)).astype("float32")
    id_col = rng.integers(0, n_subjects, (n_samples, 1))
    cat_cols = rng.integers(0, N_CATEGORIES, (n_samples, n_components))
    x_cat = np.hstack((id_col, cat_cols)).astype("float32")

    y = rng.dirichlet(np.full(n_features, 0.5), n_samples).astype("float32")

    se_idx = [0]
    id_covariate = x_num.shape[1]
    cat_start = id_covariate + 1
    interactions = [[se_idx[0], cat_start + i] for i in range(n_components)]
    C = [N_CATEGORIES] * n_components

    return SimpleNamespace(
        y=y,
        x_num=x_num,
        x_cat=x_cat,
        n_features=n_features,
        n_covariates=x_num.shape[1] + x_cat.shape[1],
        n_subjects=n_subjects,
        se_idx=se_idx,
        ca_idx=[],
        bin_idx=[],
        id_covariate=id_covariate,
        interactions=interactions,
        C=C,
    )
