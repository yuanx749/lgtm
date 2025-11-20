import argparse
import importlib
import logging
import os
import pprint
import time
from pathlib import Path

import optuna

from utils import get_vars


def objective(train_func, attrs, args, trial: optuna.trial.Trial):
    if args.method == "dgbfgp":
        args.lr = trial.suggest_categorical("lr", [5e-4, 1e-3, 5e-3, 1e-2])
    else:
        args.lr = trial.suggest_categorical("lr", [5e-3, 1e-2, 5e-2])
    args.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    args.hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    print(os.environ.get("SLURM_PROCID"))
    print(f"Trial {trial.number}")
    results = train_func(attrs, args)
    print(os.environ.get("SLURM_PROCID"))
    print(f"Trial {trial.number}")
    print(trial.params)
    best_epoch = results["best_epoch"]
    best_mse = results["best_mse"]
    best_cce = results["best_cce"]
    test_mse = results["test_mse"]
    test_cce = results["test_cce"]
    run_time = results["run_time"]
    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("test_mse", test_mse)
    trial.set_user_attr("test_cce", test_cce)
    trial.set_user_attr("run_time", run_time)
    if args.method == "gp":
        jsd = results["jsd"]
        bcd = results["bcd"]
        cos = results["cos"]
        trial.set_user_attr("jsd", jsd)
        trial.set_user_attr("bcd", bcd)
        trial.set_user_attr("cos", cos)
        if "recon_tcce" in results:
            recon_tcce = results["recon_tcce"]
            trial.set_user_attr("recon_tcce", recon_tcce)
        return round(best_cce, 3), -jsd
    if args.transform is not None:
        return best_mse
    return best_cce


def tune(train_func, attrs, args, log_name=None):
    cohort = args.cohort
    task = args.task
    method = args.method
    log_dir = Path("logs") / f"{cohort}-{task}-{method}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = optuna.logging.get_logger("optuna")
    journal_dir = Path("journals")
    journal_dir.mkdir(exist_ok=True)
    if log_name is not None:
        for h in logger.handlers[::-1]:
            if isinstance(h, logging.FileHandler):
                h.close()
                logger.removeHandler(h)
        log_file = f"{log_name}.log"
        logger.addHandler(logging.FileHandler(log_dir / log_file, mode="a"))
        journal_path = str(journal_dir / f"{cohort}-{task}-{method}.log")
        lock_obj = optuna.storages.journal.JournalFileOpenLock(journal_path)
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(journal_path, lock_obj=lock_obj)
        )
    else:
        storage = None
    study_name = f"{cohort}-{task}-{method}-{log_name}"
    if args.tune:
        args.save_pred = False
        args.save_model = False
        if args.method == "gp":
            directions = ["minimize", "minimize"]
        else:
            directions = ["minimize"]
        study = optuna.create_study(
            storage=storage,
            sampler=optuna.samplers.BruteForceSampler(),
            study_name=study_name,
            load_if_exists=True,
            directions=directions,
        )
        study.optimize(
            lambda trial: objective(train_func, attrs, args, trial),
            n_jobs=1,
        )
    else:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
        )
        if args.method == "gp":
            best_trial = min(study.best_trials, key=lambda t: t.values)
        else:
            best_trial = study.best_trial
        logger.info(best_trial.params)
        logger.info(best_trial.user_attrs)
        logger.info(os.environ.get("SLURM_JOB_ID"))
        vars(args).update(best_trial.params)
    return args

if __name__ == "__main__":

    from train import train1fold, train5folds, train1

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
    parser.add_argument("--suffix", default=time.strftime("%Y%m%d%H%M"), type=str)
    parser.add_argument("--tune", action="store_true")
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

    if args.task != "dr":
        args.name += "-clr" if args.transform else ""
        args.name += f"-f{args.fold}-s{args.seed}"
        args = tune(
            train1fold,
            attrs,
            args,
            log_name=f"f{args.fold}-s{args.seed}-{args.suffix}",
        )
        if not args.tune:
            results = train1fold(attrs, args)
            pprint.pp(get_vars(args))
    else:
        args = tune(
            train5folds,
            attrs,
            args,
            log_name=f"L{args.latent_dim}-s{args.seed}-{args.suffix}",
        )
        if not args.tune:
            results = train1(attrs, args)
            pprint.pp(get_vars(args))
