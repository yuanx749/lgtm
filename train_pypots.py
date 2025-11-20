import argparse
import importlib
import pprint
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from pypots.imputation import BRITS, SAITS
from pypots.nn.modules.loss import MSE, Criterion
from pypots.optim import Adam
from pypots.utils.random import set_random_seed

from data import split_impute, split_forecast
from utils import clr, get_vars, multi_replace, nansum
from utils2 import calc_mse, calc_cce

class CCE(Criterion):
    def __init__(self, masked=False):
        super().__init__()
        self.masked = masked

    def forward(self, logits, targets, masks=None):
        C = targets.shape[-1]
        logits = logits.view(-1, C)
        targets = targets.view(-1, C)
        value = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        if self.masked:
            masks = masks.view(-1, C).bool()
            masks = masks[:, 0]
            return torch.sum(value * masks) / (torch.sum(masks) + 1e-12)
        return value.mean()


class MSELogits(Criterion):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, masks=None):
        predictions = torch.softmax(logits, dim=-1)
        value = calc_mse(predictions, targets, masks)
        return value


def train1fold(attrs, args):
    n_features = attrs.n_features
    n_steps = attrs.n_steps
    y_3d = attrs.y_3d
    idx_2d = attrs.idx_2d
    df_y = attrs.df_y
    samples = attrs.samples

    seed = args.seed
    task = args.task
    method = args.method
    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
    transform = args.transform
    val_split = args.val_split
    fold = args.fold

    set_random_seed(seed)
    if method == "brits":
        optimizer = Adam(lr=lr)
        model = BRITS(
            n_steps=n_steps,
            n_features=n_features,
            rnn_hidden_size=hidden_dim,
            batch_size=batch_size,
            epochs=epochs,
            optimizer=optimizer,
            training_loss=CCE(masked=True) if transform is None else MSE,
            validation_metric=CCE(masked=True) if transform is None else MSE,
        )
    elif method == "saits":
        n_heads = 4
        optimizer = Adam(lr=lr)
        model = SAITS(
            n_steps=n_steps,
            n_features=n_features,
            n_layers=2,
            d_model=hidden_dim,
            n_heads=n_heads,
            d_k=hidden_dim // n_heads,
            d_v=hidden_dim // n_heads,
            d_ffn=hidden_dim,
            batch_size=batch_size,
            epochs=epochs,
            optimizer=optimizer,
            training_loss=CCE(masked=True) if transform is None else MSE,
            validation_metric=CCE(masked=True) if transform is None else MSE,
        )

    if task == "impute":
        splits = split_impute(idx_2d, val_split, random_state=0)
    elif task == "forecast":
        splits = split_forecast(idx_2d, n_steps // 2, val_split, random_state=0)
    (train_idx, val_idx, test_idx) = splits[fold]
    if transform == "clr":
        df_y_clr = df_y.copy()
        df_y_clr.loc[:] = clr(multi_replace(df_y.to_numpy()))
        y_clr_3d = df_y_clr.to_numpy().reshape(y_3d.shape)
        df_train = df_y_clr.copy()
        df_val = df_y_clr.copy()
        df_test = df_y_clr.copy()
    else:
        df_train = df_y.copy()
        df_val = df_y.copy()
        df_test = df_y.copy()
    train_samples = samples[train_idx]
    val_samples = samples[val_idx]
    test_samples = samples[test_idx]
    df_train.loc[~df_y.index.isin(train_samples)] = np.nan
    y_train = df_train.to_numpy().reshape(y_3d.shape)
    df_val.loc[val_samples.append(test_samples)] = np.nan
    y_val = df_val.to_numpy().reshape(y_3d.shape)
    df_test.loc[test_samples] = np.nan
    y_test = df_test.to_numpy().reshape(y_3d.shape)
    y_val_ori = y_test if args.val_split else y_3d
    start_time = time.time()
    model.fit({"X": y_train}, val_set={"X": y_val, "X_ori": y_val_ori})
    run_time = time.time() - start_time
    imputation = model.impute({"X": y_val})
    indicating_mask = np.isnan(y_val) ^ np.isnan(y_val_ori)
    if transform is not None:
        vmse = calc_mse(
            np.squeeze(imputation), np.nan_to_num(y_clr_3d), indicating_mask
        )
    imputation = softmax(imputation, axis=-1)
    if transform is None:
        vmse = calc_mse(np.squeeze(imputation), np.nan_to_num(y_3d), indicating_mask)
    vcce = calc_cce(np.squeeze(imputation), np.nan_to_num(y_3d), indicating_mask)
    print(f"Epoch: {model.best_epoch}")
    print(f"Val MSE: {vmse}")
    print(f"Val CCE: {vcce}")
    imputation = model.impute({"X": y_val})
    indicating_mask = np.isnan(y_test) ^ np.isnan(y_3d)
    if transform is not None:
        tmse = calc_mse(
            np.squeeze(imputation), np.nan_to_num(y_clr_3d), indicating_mask
        )
    imputation = softmax(imputation, axis=-1)
    if transform is None:
        tmse = calc_mse(np.squeeze(imputation), np.nan_to_num(y_3d), indicating_mask)
    tcce = calc_cce(np.squeeze(imputation), np.nan_to_num(y_3d), indicating_mask)
    print(f"Test MSE: {tmse}")
    print(f"Test CCE: {tcce}")

    results = dict(
        model=model,
        imputation=imputation,
        indicating_mask=indicating_mask,
        run_time=run_time,
        best_epoch=model.best_epoch,
        best_mse=vmse,
        best_cce=vcce,
        test_mse=tmse,
        test_cce=tcce,
    )

    if args.save:
        y_impute_3d = np.full(y_3d.shape, np.nan)
        y_impute_3d[indicating_mask] = imputation[indicating_mask]
        np.save(Path("output") / f"{args.name}.npy", y_impute_3d)
    return results


def train5folds(attrs, args):
    y_3d = attrs.y_3d

    n_folds = 5
    y_impute_3d = np.full((n_folds,) + y_3d.shape, np.nan)
    best_epoch_lst = []
    tmse_lst = []
    tcce_lst = []

    args.val_split = False
    for fold in range(n_folds):
        args.fold = fold
        results = train1fold(attrs, args)
        imputation = results["imputation"]
        indicating_mask = results["indicating_mask"]
        best_epoch = results["best_epoch"]
        tmse = results["test_mse"]
        tcce = results["test_cce"]
        y_impute_3d[fold, indicating_mask] = imputation[indicating_mask]
        best_epoch_lst.append(best_epoch)
        tmse_lst.append(tmse)
        tcce_lst.append(tcce)

    best_epoch = np.mean(best_epoch_lst)
    vmse = tmse = np.mean(tmse_lst)
    vcce = tcce = np.mean(tcce_lst)
    results = dict(
        best_epoch=best_epoch,
        best_mse=vmse,
        best_cce=vcce,
        test_mse=tmse,
        test_cce=tcce,
    )

    return results

class Args:
    cohort = "dhaka"
    task = "impute"
    method = "saits"
    lr = 0.001
    batch_size = 32
    hidden_dim = 64
    epochs = 100
    transform = None
    val_split = True
    fold = 4
    seed = 42
    save = False

if __name__ == "__main__":

    args = Args()

    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default=args.cohort, type=str)
    parser.add_argument("--task", default=args.task, type=str)
    parser.add_argument("--method", default=args.method, type=str)
    parser.add_argument("--fold", default=args.fold, type=int)
    parser.add_argument("--seed", default=args.seed, type=int)
    parser.add_argument("--lr", default=args.lr, type=float)
    parser.add_argument("--batch_size", default=args.batch_size, type=int)
    parser.add_argument("--hidden_dim", default=args.hidden_dim, type=int)
    namespace = parser.parse_args()
    vars(args).update(namespace.__dict__)

    args.name = f"{args.cohort}-{args.task}-{args.method}"

    attrs = importlib.import_module(f"{args.cohort}_data")

    results = train1fold(attrs, args)
    pprint.pp(get_vars(args))
