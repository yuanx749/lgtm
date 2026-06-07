import argparse
import copy
import gc
import importlib
import pprint
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from lgtm.data import MetagenomeDataset, split_forecast, split_impute
from lgtm.model import DGBFGP
from lgtm.utils import (
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
    metric_names = ["loss", "mse", "cce", "kle", "klg"]
    meters = {name: 0.0 for name in metric_names}
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
        batch_size = data.size(0)
        mse_metric = mse.view(-1, k).mean(dim=1).mean()
        cce_metric = cce.view(-1, k).mean(dim=1).mean()
        metrics_batch = {
            "loss": loss.item(),
            "mse": mse_metric.item(),
            "cce": cce_metric.item(),
            "kle": kl_qy_px.item(),
            "klg": kl_x.item(),
        }
        for name in metric_names:
            meters[name] += metrics_batch[name] * batch_size
        pred.append(torch.squeeze(pred_y, 1).detach().cpu().numpy())
    metrics = {name: meters[name] / num_data for name in metric_names}
    pred = np.concatenate(pred, axis=0)
    jsd = pw_jsd_min(model.decoder.get_loadings()) if args.linear_decoded else 0
    cos = pw_cos_min(model.decoder.get_loadings()) if args.linear_decoded else 0
    bcd = pw_bcd_min(model.decoder.get_loadings()) if args.linear_decoded else 0
    parts = [
        part
        for include, part in [
            (True, f"- Loss: {metrics['loss']:.3f}"),
            (True, f"MSE: {metrics['mse']:.3e}"),
            (True, f"CCE: {metrics['cce']:.3f}"),
            (args.encode_y, f"KLE: {metrics['kle']:.3f}"),
            (True, f"KLG: {metrics['klg']:.3f}"),
            (args.linear_decoded, f"JSD: {jsd:.3f}"),
        ]
        if include
    ]
    if b != 0:
        print(" - ".join(parts))
    metrics_out = {
        "total_loss": metrics["loss"],
        "mse": metrics["mse"],
        "cce": metrics["cce"],
        "klg": metrics["klg"],
    }
    if args.encode_y:
        metrics_out["kle"] = metrics["kle"]
    if args.linear_decoded:
        metrics_out["jsd"] = jsd
        metrics_out["cos"] = cos
        metrics_out["bcd"] = bcd
    results = {
        "metrics": metrics_out,
        "artifacts": {
            "pred": pred,
        },
    }
    return results


def train_val_test(
    attrs,
    args,
    train_set,
    val_set=None,
    test_set=None,
):

    seed = args.seed
    n_epoch = args.n_epoch
    batch_size = args.batch_size
    patience = args.patience

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
        b_init = nndsvd_init(y_train, args.latent_dim, random_state=seed)
        b_init = b_init.astype("float32")

    model_kwargs = {
        "y_num_dim": attrs.n_features,
        "x_num_dim": attrs.n_covariates,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "P": attrs.n_subjects,
        "id_embed_dim": args.id_embed_dim,
        "id_handler": args.id_handler,
        "M": args.M,
        "C": copy.deepcopy(attrs.C),
        "id_covariate": attrs.id_covariate,
        "se_idx": copy.deepcopy(attrs.se_idx),
        "ca_idx": copy.deepcopy(attrs.ca_idx),
        "bin_idx": copy.deepcopy(attrs.bin_idx),
        "interactions": copy.deepcopy(attrs.interactions),
        "k": args.k,
        "linear_decoded": args.linear_decoded,
        "non_negative": args.non_negative,
        "normalize_latent": args.normalize_latent,
        "normalize_weight": args.normalize_weight,
        "encode_y": args.encode_y,
        "b_init": b_init,
    }
    model = DGBFGP(**model_kwargs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_score = (float("inf"), float("inf"))
    best_model_weights = None
    best_epoch = n_epoch - 1
    patience_cnt = patience

    start_time = time.time()
    train_metrics = None
    val_metrics = None
    best_metrics = None
    for epoch in range(n_epoch):

        print(f"Train Epoch: {epoch} ", end="")
        train_results = train_epoch(args, train_loader, model, optimizer)
        train_metrics = train_results["metrics"]

        if val_set is not None:
            print(f"Valid Epoch: {epoch} ", end="")
            val_results = train_epoch(args, val_loader, model)
            val_metrics = val_results["metrics"]
            vmse = val_metrics["mse"]
            vcce = val_metrics["cce"]
            jsd = val_metrics.get("jsd", 0.0)
            if args.transform is not None:
                best_metric = vmse
            else:
                best_metric = vcce
            if args.linear_decoded:
                best_metric = round(best_metric, 3)
            best_metric = (best_metric, -jsd)
            if best_metric < best_score:
                best_score = best_metric
                best_model_weights = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                best_metrics = dict(val_metrics)
                patience_cnt = patience
            else:
                if args.early_stop:
                    patience_cnt -= 1
                    if patience_cnt == 0:
                        break

    run_time = time.time() - start_time

    if val_set is not None:
        model.load_state_dict(best_model_weights)

    test_metrics = None
    pred = train_results["artifacts"]["pred"] if train_metrics is not None else None
    if test_set is not None:
        print(f"Test Epoch: {best_epoch} ", end="")
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            generator=torch.Generator().manual_seed(0),
        )
        test_results = train_epoch(args, test_loader, model)
        test_metrics = test_results["metrics"]
        pred = test_results["artifacts"]["pred"]

    metrics = {"train": train_metrics}
    if best_metrics is not None:
        metrics["best"] = best_metrics
    if test_metrics is not None:
        metrics["test"] = test_metrics

    results = {
        "metrics": metrics,
        "artifacts": {
            "model": model,
            "pred": pred,
        },
        "meta": {
            "run_time": run_time,
            "best_epoch": best_epoch,
            "best_score": best_score,
        },
    }
    print(f"Time: {run_time}")

    return results


def train1fold(attrs, args):
    idx_2d = attrs.idx_2d
    df_y = attrs.df_y
    y = attrs.y

    task = args.task
    val_split = args.val_split

    if task == "impute" or task == "dr":
        splits = split_impute(idx_2d, val_split, random_state=0)
    elif task == "forecast":
        splits = split_forecast(idx_2d, attrs.n_steps // 2, val_split, random_state=0)

    train_idx, val_idx, test_idx = splits[args.fold]

    dataset = MetagenomeDataset(y, attrs.x_num, attrs.x_cat, train_idx, args.transform)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    results = train_val_test(attrs, args, train_set, val_set, test_set)
    results["artifacts"]["dataset"] = dataset

    if task == "dr" and args.encode_y:
        recon_metrics, _ = recon_opt(
            results["artifacts"]["model"],
            test_set,
            args,
        )
        results["metrics"]["recon"] = recon_metrics

    if args.save_pred:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        y_impute = np.full_like(y, np.nan)
        y_impute[test_idx] = results["artifacts"]["pred"]
        df_y_na = pd.DataFrame(np.nan, index=df_y.index, columns=df_y.columns)
        df_y_na.loc[attrs.samples[test_idx]] = y_impute[test_idx]
        y_impute_3d = df_y_na.values.reshape(attrs.y_3d.shape)
        np.save(output_dir / f"{args.name}.npy", y_impute_3d)

    return results


def train5folds(attrs, args):
    idx_2d = attrs.idx_2d

    task = args.task

    folds = {
        "best": defaultdict(list),
        "test": defaultdict(list),
        "recon": defaultdict(list),
        "meta": defaultdict(list),
    }

    args.val_split = val_split = False
    if task == "impute" or task == "dr":
        splits = split_impute(idx_2d, val_split, random_state=0)
    elif task == "forecast":
        splits = split_forecast(idx_2d, attrs.n_steps // 2, val_split, random_state=0)

    for fold, (train_idx, val_idx, test_idx) in enumerate(splits):

        args.fold = fold
        results = train1fold(attrs, args)

        for key, value in results["meta"].items():
            folds["meta"][key].append(value)

        best_metrics = results["metrics"].get("best", {})
        test_metrics = results["metrics"].get("test", {})
        recon_metrics = results["metrics"].get("recon", {})

        for key, value in best_metrics.items():
            folds["best"][key].append(value)
        for key, value in test_metrics.items():
            folds["test"][key].append(value)
        for key, value in recon_metrics.items():
            folds["recon"][key].append(value)

    metrics = {
        "folds": {
            "best": dict(folds["best"]),
            "test": dict(folds["test"]),
            "recon": dict(folds["recon"]),
        },
    }

    if folds["best"]:
        metrics["best"] = {key: np.mean(v).item() for key, v in folds["best"].items()}
    if folds["test"]:
        metrics["test"] = {key: np.mean(v).item() for key, v in folds["test"].items()}
    if folds["recon"]:
        metrics["recon"] = {key: np.mean(v).item() for key, v in folds["recon"].items()}

    meta = {"folds": dict(folds["meta"])}
    for key, v in folds["meta"].items():
        if isinstance(v[0], tuple):
            meta[key] = tuple(
                np.mean([x[i] for x in v]).item() for i in range(len(v[0]))
            )
        else:
            meta[key] = np.mean(v).item()

    results = {
        "metrics": metrics,
        "meta": meta,
    }

    return results


def train1(attrs, args):

    dataset = MetagenomeDataset(
        attrs.y, attrs.x_num, attrs.x_cat, transform=args.transform
    )
    if args.early_stop:
        results = train_val_test(attrs, args, dataset, dataset, dataset)
    else:
        results = train_val_test(attrs, args, dataset, test_set=dataset)
    results["artifacts"]["dataset"] = dataset

    if args.task == "dr" and args.encode_y:
        recon_metrics, best_state = recon_opt(
            results["artifacts"]["model"],
            dataset,
            args,
        )
        if best_state is not None:
            model = copy.deepcopy(results["artifacts"]["model"])
            model.load_state_dict(best_state)
            results["artifacts"]["model"] = model
        results["metrics"]["recon"] = recon_metrics

    if args.save_model:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        torch.save(results["artifacts"]["model"], output_dir / f"{args.name}.pt")
    return results


def recon_opt(model, dataset, args):
    model = copy.deepcopy(model)
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
    best_state = None
    for epoch in range(args.n_epoch // 2):
        opt_results = train_epoch(args_opt, test_loader, model, optimizer)
        tcce = opt_results["metrics"]["cce"]
        tmse = opt_results["metrics"]["mse"]
        if tcce < recon_tcce:
            recon_tcce = tcce
            recon_tmse = tmse
            best_state = copy.deepcopy(model.state_dict())
    recon_metrics = {
        "tmse": recon_tmse,
        "tcce": recon_tcce,
    }
    return recon_metrics, best_state


if __name__ == "__main__":

    cohort = "hmp"

    parser_dataset = argparse.ArgumentParser(add_help=False)
    parser_dataset.add_argument("--cohort", default=cohort, type=str)
    namespace, _ = parser_dataset.parse_known_args()
    cohort = namespace.cohort

    config = importlib.import_module("lgtm.config")
    attrs = importlib.import_module(f"scripts.{cohort}_data")
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
