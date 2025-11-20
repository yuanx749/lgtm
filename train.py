import argparse
import copy
import gc
import importlib
import pprint
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data import MetagenomeDataset, split_forecast, split_impute
from model import DGBFGP
from utils import (
    get_vars,
    pw_bcd_min,
    pw_cos_min,
    pw_jsd_min,
    pw_topic_distance,
    nndsvd_init,
)


def train_epoch(args, dataloader, model, optimizer=None):
    is_train = optimizer is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_train:
        model.train()
    else:
        model.eval()
    k = args.k
    if not is_train:
        k = 1
    b = args.b
    loss_function = args.loss_function
    num_data = len(dataloader.dataset)
    total_loss = 0
    mse_loss = 0
    kld_loss = 0
    cce_loss = 0
    klq_loss = 0
    pred = []
    for batch in tqdm(dataloader, disable=True):
        data = batch["y"].to(device)
        y_mask = batch["y_mask"].to(device)
        covar = batch["x"].to(device)
        x_mask = batch["x_mask"].to(device)
        if is_train and args.encode_y:
            (
                logits_x,
                pred_y,
                log_theta_x,
                densities,
                A_samples,
                logits_y,
                recon_y,
                log_theta_y,
                f_x,
            ) = model(covar, y=data, x_mask=x_mask, stochastic_flag=is_train)
            mse_y, cce_y = model.recon_loss(logits_y, recon_y, data, y_mask)
            mse_x, cce_x = model.pred_loss(logits_x, pred_y, data, y_mask)
            mse = mse_y
            cce = cce_y
        else:
            (
                logits_x,
                pred_y,
                log_theta_x,
                densities,
                A_samples,
                logits_y,
                recon_y,
                log_theta_y,
                f_x,
            ) = model(covar, x_mask=x_mask, stochastic_flag=is_train)
            mse_y = cce_y = torch.zeros(1, device=device)
            if args.transform is None:
                mse_x, cce_x = model.pred_loss(logits_x, pred_y, data, y_mask)
            else:
                mse_x, cce_x = model.pred_loss_clr(logits_x, data, y_mask)
            mse = mse_x
            cce = cce_x

        if is_train and args.encode_y:
            kl_qy_px = model.kl_loss_qy_px(log_theta_x, log_theta_y)
        else:
            kl_qy_px = torch.zeros(1, device=device)
        kl_x = model.klx_loss(densities)
        kl = b * kl_x
        if args.encode_y and b != 0:
            kl = kl + kl_qy_px

        if loss_function == "mse":
            loss = mse.mean() + kl
        elif loss_function == "cce":
            loss = cce.mean() + kl

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(data) * k
        mse_loss += mse.sum().item()
        cce_loss += cce.sum().item()
        kld_loss += kl_x.item() * len(data)
        klq_loss += kl_qy_px.item() * len(data)
        pred.append(torch.squeeze(pred_y, 1).detach().cpu().numpy())
    total_loss = total_loss / num_data / k
    mse_loss = mse_loss / num_data / k
    cce_loss = cce_loss / num_data / k
    kld_loss = kld_loss / num_data
    klq_loss = klq_loss / num_data
    pred = np.concatenate(pred, axis=0)
    jsd = pw_jsd_min(model.decoder.get_loadings()) if args.linear_decoded else 0
    cos = pw_cos_min(model.decoder.get_loadings()) if args.linear_decoded else 0
    bcd = pw_bcd_min(model.decoder.get_loadings()) if args.linear_decoded else 0
    if b != 0:
        print(
            f"- Loss: {total_loss:.3f} - MSE: {mse_loss:.3e} - CCE: {cce_loss:.3f} - JSD: {jsd:.3f} - KL1: {klq_loss:.3f} - KL2: {kld_loss:.3f}"
        )
    results = dict(
        total_loss=total_loss,
        mse_loss=mse_loss,
        cce_loss=cce_loss,
        pred=pred,
        jsd=jsd,
        cos=cos,
        bcd=bcd,
    )
    return results


def train_val_test(
    attrs,
    args,
    train_set,
    val_set=None,
    test_set=None,
):
    n_features = attrs.n_features
    n_subjects = attrs.n_subjects
    n_covariates = attrs.n_covariates

    seed = args.seed
    y_num_dim = n_features
    x_num_dim = n_covariates
    id_embed_dim = args.id_embed_dim
    P = n_subjects
    M = args.M
    latent_dim = args.latent_dim
    se_idx = copy.deepcopy(args.se_idx)
    ca_idx = copy.deepcopy(args.ca_idx)
    bin_idx = copy.deepcopy(args.bin_idx)
    interactions = copy.deepcopy(args.interactions)
    C = copy.deepcopy(args.C)
    id_covariate = args.id_covariate
    id_handler = args.id_handler
    k = args.k
    n_epoch = args.n_epoch
    lr = args.lr
    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
    linear_decoded = args.linear_decoded
    non_negative = args.non_negative
    normalize_latent = args.normalize_latent
    normalize_weight = args.normalize_weight
    encode_y = args.encode_y
    patience = args.patience
    early_stop = args.early_stop
    transform = args.transform

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(0),
    )
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            generator=torch.Generator().manual_seed(0),
        )

    gc.collect()
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    b_init = None
    if args.init_b:
        y_train = (
            train_set.dataset.y[train_set.indices]
            if isinstance(train_set, Subset)
            else train_set.y
        )
        b_init = nndsvd_init(y_train, latent_dim, random_state=seed)
        b_init = b_init.astype("float32")

    model = DGBFGP(
        y_num_dim,
        x_num_dim,
        latent_dim,
        hidden_dim,
        P,
        id_embed_dim,
        id_handler,
        M,
        C,
        id_covariate,
        se_idx,
        ca_idx,
        bin_idx,
        interactions,
        k=k,
        linear_decoded=linear_decoded,
        non_negative=non_negative,
        normalize_latent=normalize_latent,
        normalize_weight=normalize_weight,
        encode_y=encode_y,
        b_init=b_init,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_score = float("inf")
    if args.method == "gp":
        best_score = (float("inf"), float("inf"))
    best_loss = float("inf")
    best_mse = float("inf")
    best_cce = float("inf")
    best_model_weights = None
    best_epoch = n_epoch - 1
    best_jsd = 0
    best_cos = 0
    best_bcd = 0
    patience_cnt = patience

    start_time = time.time()
    for epoch in range(n_epoch):

        print(f"Train Epoch: {epoch} ", end="")
        train_results = train_epoch(args, train_loader, model, optimizer)

        if val_set is not None:
            print(f"Valid Epoch: {epoch} ", end="")
            val_results = train_epoch(args, val_loader, model)
            valid_loss = val_results["total_loss"]
            vmse_loss = val_results["mse_loss"]
            vcce_loss = val_results["cce_loss"]
            jsd = val_results["jsd"]
            cos = val_results["cos"]
            bcd = val_results["bcd"]
            if transform is not None:
                best_metric = vmse_loss
            else:
                best_metric = vcce_loss
            if args.method == "gp":
                best_metric = (round(vcce_loss, 3), -jsd)
            if best_metric < best_score:
                if early_stop:
                    best_score = best_metric
                best_loss = valid_loss
                best_mse = vmse_loss
                best_cce = vcce_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                best_jsd = jsd
                best_cos = cos
                best_bcd = bcd
                patience_cnt = patience
            else:
                patience_cnt -= 1
                if patience_cnt == 0:
                    break
    run_time = time.time() - start_time

    if val_set is not None:
        model.load_state_dict(best_model_weights)
    results = dict(
        model=model,
        run_time=run_time,
        best_epoch=best_epoch,
        best_mse=best_mse,
        best_cce=best_cce,
        jsd=best_jsd,
        cos=best_cos,
        bcd=best_bcd,
    )

    if test_set is not None:
        print(f"Test Epoch: {best_epoch} ", end="")
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            generator=torch.Generator().manual_seed(0),
        )
        test_results = train_epoch(args, test_loader, model)
        test_mse = test_results["mse_loss"]
        test_cce = test_results["cce_loss"]
        pred = test_results["pred"]
        results.update(
            test_mse=test_mse,
            test_cce=test_cce,
            pred=pred,
        )
    print(f"Time: {run_time}")

    return results


def train1fold(attrs, args):
    n_steps = attrs.n_steps
    x_num = attrs.x_num
    x_cat = attrs.x_cat
    y = attrs.y
    y_3d = attrs.y_3d
    idx_2d = attrs.idx_2d
    df_y = attrs.df_y
    samples = attrs.samples

    task = args.task
    val_split = args.val_split
    transform = args.transform
    fold = args.fold

    if task == "impute" or task == "dr":
        splits = split_impute(idx_2d, val_split, random_state=0)
    elif task == "forecast":
        splits = split_forecast(idx_2d, n_steps // 2, val_split, random_state=0)

    train_idx, val_idx, test_idx = splits[fold]

    dataset = MetagenomeDataset(y, x_num, x_cat, train_idx, transform)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    results = train_val_test(attrs, args, train_set, val_set, test_set)
    results.update(dataset=dataset)

    if task == "dr" and args.encode_y:
        model = copy.deepcopy(results["model"])
        model.decoder.requires_grad_(False)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(0),
        )
        args_opt = copy.deepcopy(args)
        args_opt.b = 0
        recon_tcce = float("inf")
        recon_tmse = float("inf")
        for epoch in range(args.n_epoch // 2):
            opt_results = train_epoch(args_opt, test_loader, model, optimizer)
            tmse_loss = opt_results["mse_loss"]
            tcce_loss = opt_results["cce_loss"]
            if tcce_loss < recon_tcce:
                recon_tcce = tcce_loss
                recon_tmse = tmse_loss
        results.update(
            recon_tmse=recon_tmse,
            recon_tcce=recon_tcce,
        )

    if args.save_pred:
        y_impute = np.full_like(y, np.nan)
        y_impute[test_idx] = results["pred"]
        df_y_na = pd.DataFrame(np.nan, index=df_y.index, columns=df_y.columns)
        df_y_na.loc[samples[test_idx]] = y_impute[test_idx]
        y_impute_3d = df_y_na.values.reshape(y_3d.shape)
        np.save(Path("output") / f"{args.name}.npy", y_impute_3d)

    return results


def train5folds(attrs, args):
    n_steps = attrs.n_steps
    y = attrs.y
    y_3d = attrs.y_3d
    idx_2d = attrs.idx_2d
    df_y = attrs.df_y
    samples = attrs.samples

    task = args.task

    n_folds = 5
    y_impute = np.full_like(y, np.nan)
    y_impute_3d = np.full((n_folds,) + y_3d.shape, np.nan)
    run_time_lst = []
    best_epoch_lst = []
    best_mse_lst = []
    best_cce_lst = []
    test_mse_lst = []
    test_cce_lst = []
    jsd_lst = []
    cos_lst = []
    bcd_lst = []
    test_cce_opt_lst = []
    test_mse_opt_lst = []

    args.val_split = val_split = False
    if task == "impute" or task == "dr":
        splits = split_impute(idx_2d, val_split, random_state=0)
    elif task == "forecast":
        splits = split_forecast(idx_2d, n_steps // 2, val_split, random_state=0)

    for fold, (train_idx, val_idx, test_idx) in enumerate(splits):

        args.fold = fold
        results = train1fold(attrs, args)
        model = results["model"]
        run_time = results["run_time"]
        best_epoch = results["best_epoch"]
        best_mse = results["best_mse"]
        best_cce = results["best_cce"]
        test_mse = results["test_mse"]
        test_cce = results["test_cce"]

        run_time_lst.append(run_time)
        best_epoch_lst.append(best_epoch)
        best_mse_lst.append(best_mse)
        best_cce_lst.append(best_cce)
        test_mse_lst.append(test_mse)
        test_cce_lst.append(test_cce)
        if args.linear_decoded:
            B = model.decoder.get_loadings()
            jsd = pw_jsd_min(B)
            cos = pw_cos_min(B)
            bcd = pw_bcd_min(B)
            jsd_lst.append(jsd)
            cos_lst.append(cos)
            bcd_lst.append(bcd)

        y_impute[test_idx] = results["pred"]
        df_y_na = pd.DataFrame(np.nan, index=df_y.index, columns=df_y.columns)
        df_y_na.loc[samples[test_idx]] = y_impute[test_idx]
        y_impute_3d[fold] = df_y_na.values.reshape(y_3d.shape)

        if task == "dr" and args.encode_y:
            recon_tmse = results["recon_tmse"]
            recon_tcce = results["recon_tcce"]
            test_mse_opt_lst.append(recon_tmse)
            test_cce_opt_lst.append(recon_tcce)

    run_time = np.mean(run_time_lst)
    best_epoch = np.mean(best_epoch_lst)
    best_mse = test_mse = np.mean(test_mse_lst)
    best_cce = test_cce = np.mean(test_cce_lst)
    results = dict(
        run_time=run_time,
        best_epoch=best_epoch,
        best_mse=best_mse,
        best_cce=best_cce,
        test_mse=test_mse,
        test_cce=test_cce,
    )
    if args.linear_decoded:
        jsd = np.mean(jsd_lst)
        cos = np.mean(cos_lst)
        bcd = np.mean(bcd_lst)
        results.update(
            jsd=jsd,
            cos=cos,
            bcd=bcd,
        )
    if task == "dr" and args.encode_y:
        recon_tmse = np.mean(test_mse_opt_lst)
        recon_tcce = np.mean(test_cce_opt_lst)
        results.update(
            recon_tmse=recon_tmse,
            recon_tcce=recon_tcce,
        )

    return results


def train1(attrs, args):
    x_num = attrs.x_num
    x_cat = attrs.x_cat
    y = attrs.y
    transform = args.transform

    dataset = MetagenomeDataset(y, x_num, x_cat, transform=transform)
    if args.early_stop:
        results = train_val_test(attrs, args, dataset, dataset, dataset)
    else:
        results = train_val_test(attrs, args, dataset, test_set=dataset)
    results.update(dataset=dataset)

    if args.task == "dr" and args.encode_y:
        model = copy.deepcopy(results["model"])
        model.decoder.requires_grad_(False)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        test_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(0),
        )
        args_opt = copy.deepcopy(args)
        args_opt.b = 0
        recon_tcce = float("inf")
        recon_tmse = float("inf")
        for epoch in range(args.n_epoch // 2):
            opt_results = train_epoch(args_opt, test_loader, model, optimizer)
            tcce_loss = opt_results["cce_loss"]
            tmse_loss = opt_results["mse_loss"]
            if tcce_loss < recon_tcce:
                recon_tcce = tcce_loss
                recon_tmse = tmse_loss
                best_model_weights = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_model_weights)
        results.update(
            recon_tmse=recon_tmse,
            recon_tcce=recon_tcce,
            model=model,
        )

    if args.save_model:
        torch.save(results["model"], Path("output") / f"{args.name}.pt")
    return results


if __name__ == "__main__":

    cohort = "hmp"

    parser_dataset = argparse.ArgumentParser(add_help=False)
    parser_dataset.add_argument("--cohort", default=cohort, type=str)
    namespace, _ = parser_dataset.parse_known_args()
    cohort = namespace.cohort

    config = importlib.import_module("config")
    attrs = importlib.import_module(f"{cohort}_data")
    args = getattr(config, "args")

    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default=cohort, type=str)
    parser.add_argument("--task", default=args.task, type=str)
    parser.add_argument("--method", default=args.method, type=str)
    parser.add_argument("--latent_dim", default=args.latent_dim, type=int)
    parser.add_argument("--fold", default=args.fold, type=int)
    parser.add_argument("--seed", default=args.seed, type=int)
    namespace = parser.parse_args()
    vars(args).update(namespace.__dict__)

    if args.task != "dr":
        args.val_split = True

    if args.method == "dgbfgp":
        args.normalize_latent = None
        args.linear_decoded = False
        args.non_negative = False
        args.normalize_weight = False
        args.encode_y = False
        args.init_b = False

    args.name = f"{args.cohort}-{args.task}-{args.method}"
    args.name += f"-L{args.latent_dim}"

    results = train1(attrs, args)
    pprint.pp(get_vars(args))
