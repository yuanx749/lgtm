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

from data import MetagenomeDataset, split_impute, split_forecast
from model_simple import SimpleVAE
from utils import get_vars, pw_cos_min, pw_jsd_min, pw_bcd_min, pw_topic_distance


def train_epoch(args, dataloader, model, optimizer=None):
    is_train = optimizer is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_train:
        model.train()
    else:
        model.eval()
    b = args.b
    num_data = len(dataloader.dataset)
    total_loss = 0
    mse_loss = 0
    kld_loss = 0
    cce_loss = 0
    pred = []
    for batch in tqdm(dataloader, disable=True):
        data = batch["y"].to(device)
        logits, recon_y, mu, logvar, theta = model(data)
        loss, mse, cce, kld = model.vae_loss(logits, recon_y, mu, logvar, data, b)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(data)
        mse_loss += mse.item() * len(data)
        kld_loss += kld.item() * len(data)
        cce_loss += cce.item() * len(data)
        pred.append(torch.squeeze(recon_y, 1).detach().cpu().numpy())
    total_loss /= num_data
    mse_loss /= num_data
    kld_loss /= num_data
    cce_loss /= num_data
    pred = np.concatenate(pred, axis=0)
    jsd = pw_jsd_min(model.decoder.get_loadings()) if args.linear_decoded else 0
    cos = pw_cos_min(model.decoder.get_loadings()) if args.linear_decoded else 0
    bcd = pw_bcd_min(model.decoder.get_loadings()) if args.linear_decoded else 0
    print(
        f"Loss: {total_loss:.3e}, MSE: {mse_loss:.3e}, CCE: {cce_loss:.3e}, JSD: {jsd:.3f}, KLD: {kld_loss:.3e}"
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
    seed = args.seed
    n_epoch = args.n_epoch
    lr = args.lr
    batch_size = args.batch_size
    latent_dim = args.latent_dim
    hidden_dim = args.hidden_dim
    linear_decoded = args.linear_decoded
    non_negative = args.non_negative
    normalize_latent = args.normalize_latent
    normalize_weight = args.normalize_weight
    patience = args.patience
    early_stop = args.early_stop

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

    input_dim = n_features
    model = SimpleVAE(
        input_dim,
        latent_dim,
        hidden_dim,
        linear_decoded,
        non_negative,
        normalize_latent,
        normalize_weight,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_score = float("inf")
    best_score = (float("inf"), float("inf"))
    best_mse = float("inf")
    best_cce = float("inf")
    best_model_weights = None
    best_epoch = n_epoch - 1
    best_jsd = 0
    best_bcd = 0
    best_cos = 0
    patience_cnt = patience

    for epoch in range(n_epoch):
        print(f"Epoch: {epoch}, Train ", end="")
        train_results = train_epoch(args, train_loader, model, optimizer)

        if val_set is not None:
            print(f"Epoch: {epoch}, Validation ", end="")
            val_results = train_epoch(args, val_loader, model)
            vmse_loss = val_results["mse_loss"]
            vcce_loss = val_results["cce_loss"]
            jsd = val_results["jsd"]
            cos = val_results["cos"]
            bcd = val_results["bcd"]
            best_metric = (round(vcce_loss, 3), -jsd)
            if best_metric < best_score:
                if early_stop:
                    best_score = best_metric
                best_mse = vmse_loss
                best_cce = vcce_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                best_jsd = jsd
                best_bcd = bcd
                best_cos = cos
                patience_cnt = patience
            else:
                patience_cnt -= 1
                if patience_cnt == 0:
                    break
    if val_set is not None:
        model.load_state_dict(best_model_weights)
    results = dict(
        model=model,
        best_epoch=best_epoch,
        best_mse=best_mse,
        best_cce=best_cce,
        best_jsd=best_jsd,
        best_cos=best_cos,
        best_bcd=best_bcd,
    )

    if test_set is not None:
        print(f"Epoch: {best_epoch}, Test ", end="")
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

    return results


def train1fold(attrs, args):
    n_steps = attrs.n_steps
    x = attrs.x
    x_num = attrs.x_num
    x_cat = attrs.x_cat
    y = attrs.y
    idx_2d = attrs.idx_2d

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
        print(f"Test Epoch: {epoch} ", end="")
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

    return results


def train5folds(attrs, args):
    n_steps = attrs.n_steps
    y = attrs.y
    y_3d = attrs.y_3d
    idx_2d = attrs.idx_2d
    df_y = attrs.df_y
    samples = attrs.samples

    task = args.task
    val_split = args.val_split

    n_folds = 5
    y_impute = np.full_like(y, np.nan)
    y_impute_3d = np.full((n_folds,) + y_3d.shape, np.nan)
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

    if task == "impute" or task == "dr":
        splits = split_impute(idx_2d, val_split, random_state=0)
    elif task == "forecast":
        splits = split_forecast(idx_2d, n_steps // 2, val_split, random_state=0)

    for fold, (train_idx, val_idx, test_idx) in enumerate(splits):

        args.fold = fold
        results = train1fold(attrs, args)
        model = results["model"]
        best_epoch = results["best_epoch"]
        best_mse = results["best_mse"]
        best_cce = results["best_cce"]
        test_mse = results["test_mse"]
        test_cce = results["test_cce"]

        best_epoch_lst.append(best_epoch)
        best_mse_lst.append(best_mse)
        best_cce_lst.append(best_cce)
        test_mse_lst.append(test_mse)
        test_cce_lst.append(test_cce)
        if args.linear_decoded:
            B = model.decoder.get_loadings()
            jsd = pw_jsd_min(B)
            bcd = pw_bcd_min(B)
            cos = pw_cos_min(B)
            jsd_lst.append(jsd)
            bcd_lst.append(bcd)
            cos_lst.append(cos)

        y_impute[test_idx] = results["pred"]
        df_y_na = pd.DataFrame(np.nan, index=df_y.index, columns=df_y.columns)
        df_y_na.loc[samples[test_idx]] = y_impute[test_idx]
        y_impute_3d[fold] = df_y_na.values.reshape(y_3d.shape)

        recon_tmse = results["recon_tmse"]
        recon_tcce = results["recon_tcce"]
        test_mse_opt_lst.append(recon_tmse)
        test_cce_opt_lst.append(recon_tcce)

    best_mse = test_mse = np.mean(test_mse_lst)
    best_cce = test_cce = np.mean(test_cce_lst)
    results = dict(
        best_mse=best_mse,
        best_cce=best_cce,
        test_mse=test_mse,
        test_cce=test_cce,
    )
    if args.linear_decoded:
        jsd = np.mean(jsd_lst)
        bcd = np.mean(bcd_lst)
        cos = np.mean(cos_lst)
        results.update(
            jsd=jsd,
            bcd=bcd,
            cos=cos,
        )
    recon_tmse = np.mean(test_mse_opt_lst)
    recon_tcce = np.mean(test_cce_opt_lst)
    results.update(
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
    return results


class Args:
    cohort = "hmp"
    task = "dr"
    method = "ldavae"
    latent_dim = 6
    hidden_dim = 64
    n_epoch = 100
    patience = 0
    b = 0.001
    lr = 0.05
    batch_size = 64
    transform = None
    normalize_latent = True
    linear_decoded = True
    non_negative = True
    normalize_weight = True
    seed = 42
    fold = 4
    val_split = False
    early_stop = True


if __name__ == "__main__":

    args = Args()

    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default=args.cohort, type=str)
    parser.add_argument("--lr", default=args.lr, type=float)
    parser.add_argument("--latent_dim", default=args.latent_dim, type=int)
    parser.add_argument("--hidden_dim", default=args.hidden_dim, type=int)
    parser.add_argument("--batch_size", default=args.batch_size, type=int)
    parser.add_argument("--seed", default=args.seed, type=int)
    namespace = parser.parse_args()
    vars(args).update(namespace.__dict__)
    args.name = f"{args.cohort}-{args.task}-{args.method}"

    attrs = importlib.import_module(f"{args.cohort}_data")

    results = train1(attrs, args)
    pprint.pp(get_vars(args))
