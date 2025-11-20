import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils import clr, multi_replace


class MetagenomeDataset(Dataset):

    def __init__(self, y, x_num, x_cat, train_idx=slice(None), transform=None):
        if transform == "clr":
            self.y = clr(multi_replace(y))
        else:
            self.y = y
        self.y_mask = ~np.isnan(y)
        self.scaler = StandardScaler().fit(x_num[train_idx])
        x_num = self.scaler.transform(x_num)
        self.x = np.hstack((x_num, x_cat))
        self.x_mask = np.hstack((np.full(x_num.shape, True), x_cat != -1))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        y = self.y[idx, :]
        y = torch.Tensor(np.array(y))

        y_mask = self.y_mask[idx, :]
        y_mask = np.array(y_mask)
        y_mask = torch.from_numpy(y_mask)

        x = self.x[idx, :]
        x = torch.Tensor(np.array(x))

        x_mask = self.x_mask[idx, :]
        x_mask = np.array(x_mask)
        x_mask = torch.from_numpy(x_mask)

        sample = {
            "y": y,
            "x": x,
            "idx": idx,
            "y_mask": y_mask,
            "x_mask": x_mask,
        }
        return sample


def split_impute(idx_2d, val_split=True, random_state=None):
    n_folds = 5
    n_samples = np.nanmax(idx_2d).astype(int) + 1
    n_subjects = idx_2d.shape[0]
    subject_idx = np.argwhere(~np.isnan(idx_2d))[:, 0]
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    _, test_idx_lst = zip(*list(cv.split(range(n_samples), subject_idx)))
    splits = []
    for i in range(n_folds):
        test_idx = test_idx_lst[i]
        if val_split:
            val_idx = test_idx_lst[i - 1][::2]
        else:
            val_idx = test_idx.copy()
        train_idx = np.setdiff1d(np.arange(n_samples), np.union1d(val_idx, test_idx))
        train_subjects = np.unique(np.argwhere(np.isin(idx_2d, train_idx))[:, 0])
        assert len(train_subjects) == n_subjects
        splits.append((train_idx, val_idx, test_idx))
    return tuple(splits)


def split_forecast(idx_2d, n_future_steps, val_split=True, random_state=None):
    n_folds = 5
    n_samples = np.nanmax(idx_2d).astype(int) + 1
    n_subjects = idx_2d.shape[0]
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    _, test_subject_lst = zip(*list(cv.split(range(n_subjects))))
    past_subjects = np.unique(np.argwhere(~np.isnan(idx_2d[:, :-n_future_steps]))[:, 0])
    splits = []
    for i in range(n_folds):
        test_subjects = test_subject_lst[i]
        test_subjects = np.intersect1d(test_subjects, past_subjects)
        test_samples = idx_2d[test_subjects, -n_future_steps:]
        test_idx = test_samples[~np.isnan(test_samples)].ravel().astype(int)
        if val_split:
            val_subjects = test_subject_lst[i - 1][::2]
            val_subjects = np.intersect1d(val_subjects, past_subjects)
            val_samples = idx_2d[val_subjects, -n_future_steps:]
            val_idx = val_samples[~np.isnan(val_samples)].ravel().astype(int)
        else:
            val_idx = test_idx.copy()
        train_idx = np.setdiff1d(np.arange(n_samples), np.union1d(val_idx, test_idx))
        train_subjects = np.unique(np.argwhere(np.isin(idx_2d, train_idx))[:, 0])
        assert len(train_subjects) == n_subjects
        splits.append((train_idx, val_idx, test_idx))
    return tuple(splits)
