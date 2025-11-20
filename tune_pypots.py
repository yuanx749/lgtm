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
    args.lr = trial.suggest_categorical("lr", [5e-4, 1e-3, 5e-3, 1e-2])
    args.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    args.hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    print(os.environ.get("SLURM_PROCID"))
    print(f"Trial {trial.number}")
    results = train_func(attrs, args)
    print(os.environ.get("SLURM_PROCID"))
    print(f"Trial {trial.number}")
    print(trial.params)
    best_epoch = results["best_epoch"]
    vmse = results["best_mse"]
    vcce = results["best_cce"]
    tmse = results["test_mse"]
    tcce = results["test_cce"]
    run_time = results["run_time"]
    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("test_mse", tmse)
    trial.set_user_attr("test_cce", tcce)
    trial.set_user_attr("run_time", run_time)
    if args.transform is not None:
        return vmse
    return vcce


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
        args.save = False
        study = optuna.create_study(
            storage=storage,
            sampler=optuna.samplers.BruteForceSampler(),
            study_name=study_name,
            direction="minimize",
            load_if_exists=True,
        )
        study.optimize(
            lambda trial: objective(train_func, attrs, args, trial),
            n_jobs=1,
        )
    else:
        args.save = True
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
        )
        best_trial = study.best_trial
        logger.info(best_trial.params)
        logger.info(best_trial.user_attrs)
        logger.info(os.environ.get("SLURM_JOB_ID"))
        vars(args).update(best_trial.params)
    return args

if __name__ == "__main__":

    from train_pypots import train1fold

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
        save = True

    args = Args()

    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default=args.cohort, type=str)
    parser.add_argument("--task", default=args.task, type=str)
    parser.add_argument("--method", default=args.method, type=str)
    parser.add_argument("--fold", default=args.fold, type=int)
    parser.add_argument("--seed", default=args.seed, type=int)
    parser.add_argument("--suffix", default=time.strftime("%Y%m%d%H%M"), type=str)
    parser.add_argument("--tune", action="store_true")
    namespace = parser.parse_args()
    vars(args).update(namespace.__dict__)

    args.name = f"{args.cohort}-{args.task}-{args.method}"
    args.name += "-clr" if args.transform else ""
    args.name += f"-f{args.fold}-s{args.seed}"

    attrs = importlib.import_module(f"{args.cohort}_data")

    args = tune(
        train1fold,
        attrs,
        args,
        log_name=f"f{args.fold}-s{args.seed}-{args.suffix}",
    )
    if not args.tune:
        results = train1fold(attrs, args)
        pprint.pp(get_vars(args))
