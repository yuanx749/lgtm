import numpy as np
from sklearn.decomposition import NMF, LatentDirichletAllocation


class Mean:

    def __init__(self, axis=(0, 1)):
        self.axis = axis

    def impute(self, test_set):
        X = test_set["X"]
        if isinstance(X, list):
            X = np.asarray(X)
        assert len(X.shape) == 3, (
            f"Input X should have 3 dimensions [n_subjects, n_steps, n_features], "
            f"but the actual shape of X: {X.shape}"
        )
        if isinstance(X, np.ndarray):
            mean_values = np.nanmean(X, axis=self.axis, keepdims=True)
            mean_values = np.nan_to_num(mean_values)
            imputed_data = np.where(np.isnan(X), mean_values, X)
        else:
            raise ValueError()
        return imputed_data


class Interp:

    def impute(self, test_set):
        X = test_set["X"]
        if isinstance(X, list):
            X = np.asarray(X)
        assert len(X.shape) == 3, (
            f"Input X should have 3 dimensions [n_subjects, n_steps, n_features], "
            f"but the actual shape of X: {X.shape}"
        )

        def _interpolate_missing_values(X: np.ndarray):
            nans = np.isnan(X)
            nan_index = np.where(nans)[0]
            index = np.where(~nans)[0]
            if np.any(nans) and index.size > 0:
                X[nans] = np.interp(nan_index, index, X[~nans])
            elif np.any(nans):
                X[nans] = 0

        if isinstance(X, np.ndarray):
            trans_X = X.transpose((0, 2, 1))
            n_samples, n_features, n_steps = trans_X.shape
            reshaped_X = np.reshape(trans_X, (-1, n_steps))
            imputed_X = np.ones(reshaped_X.shape)
            for i, univariate_series in enumerate(reshaped_X):
                t = np.copy(univariate_series)
                _interpolate_missing_values(t)
                imputed_X[i] = t
            imputed_trans_X = np.reshape(imputed_X, (n_samples, n_features, -1))
            imputed_data = imputed_trans_X.transpose((0, 2, 1))
        else:
            raise ValueError()
        return imputed_data


def calc_mse(predictions, targets, masks):
    if masks is not None:
        return np.sum(np.square(predictions - targets) * masks) / np.sum(masks)
    else:
        return np.mean(np.square(predictions - targets))


def calc_cce(predictions, targets, masks):
    C = targets.shape[-1]
    predictions = predictions.reshape(-1, C)
    targets = targets.reshape(-1, C)
    value = np.nansum(-targets * np.log(predictions + 1e-10), axis=-1)
    if masks is not None:
        masks = masks.reshape(-1, C)
        masks = masks[:, 0]
        return np.sum(value * masks) / np.sum(masks)
    return np.mean(value)


class NNMF(NMF):

    def __init__(
        self,
        n_components="auto",
        *,
        init=None,
        solver="cd",
        beta_loss="frobenius",
        tol=1e-4,
        max_iter=200,
        random_state=None,
        alpha_W=0.0,
        alpha_H="same",
        l1_ratio=0.0,
        verbose=0,
        shuffle=False,
    ):
        super().__init__(
            n_components,
            init=init,
            solver=solver,
            beta_loss=beta_loss,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            l1_ratio=l1_ratio,
            verbose=verbose,
            shuffle=shuffle,
        )

    def transform(self, X, *, normalize=True):
        W = super().transform(X)
        if normalize:
            W = W / W.sum(axis=1, keepdims=True)
        return W

    def get_components(self, *, normalize=True):
        H = self.components_
        if normalize:
            H = H / H.sum(axis=1, keepdims=True)
        return H

class LDA(LatentDirichletAllocation):

    def __init__(
        self,
        n_components=10,
        *,
        doc_topic_prior=None,
        topic_word_prior=None,
        learning_method="batch",
        learning_decay=0.7,
        learning_offset=10.0,
        max_iter=10,
        batch_size=128,
        evaluate_every=-1,
        total_samples=1e6,
        perp_tol=1e-1,
        mean_change_tol=1e-3,
        max_doc_update_iter=100,
        n_jobs=None,
        verbose=0,
        random_state=None,
        total_count=1e6,
    ):
        super().__init__(
            n_components,
            doc_topic_prior=doc_topic_prior,
            topic_word_prior=topic_word_prior,
            learning_method=learning_method,
            learning_decay=learning_decay,
            learning_offset=learning_offset,
            max_iter=max_iter,
            batch_size=batch_size,
            evaluate_every=evaluate_every,
            total_samples=total_samples,
            perp_tol=perp_tol,
            mean_change_tol=mean_change_tol,
            max_doc_update_iter=max_doc_update_iter,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
        )
        self.total_count = total_count

    def fit(self, X, y = None):
        X = X * self.total_count
        return super().fit(X, y)

    def transform(self, X, *, normalize=True):
        X = X * self.total_count
        Z = super().transform(X, normalize=normalize)
        return Z

    def get_components(self, *, normalize=True):
        B = self.components_
        if normalize:
            B = B / B.sum(axis=1, keepdims=True)
        return B
