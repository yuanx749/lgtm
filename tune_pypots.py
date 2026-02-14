import argparse
import importlib
import json
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
    tmse = results["test_mse"]
    tcce = results["test_cce"]
    run_time = results["run_time"]
    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("run_time", run_time)
    trial.set_user_attr("test_mse", tmse)
    trial.set_user_attr("test_cce", tcce)
    if args.transform is not None:
        return results["best_mse"]
    return results["best_cce"]


def tune(train_func, attrs, args, log_name=None):
    cohort = args.cohort
    task = args.task
    method = args.method
    log_dir = Path("logs") / f"{cohort}-{task}-{method}"
    log_dir.mkdir(parents=True, exist_ok=True)
    journal_dir = Path("journals")
    journal_dir.mkdir(exist_ok=True)
    if log_name is not None:
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
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
        )
        best_trial = study.best_trial
        vars(args).update(best_trial.params)
    return study


def write_summary(args, study, log_name):
    best_trial = study.best_trial
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_submit_dir = os.environ.get("SLURM_SUBMIT_DIR")
    slurm_stdout = None
    if slurm_job_id and slurm_submit_dir:
        slurm_stdout = str(
            Path(slurm_submit_dir) / "slurm" / f"slurm-{slurm_job_id}.out"
        )
    summary = {
        "study": {
            "name": study.study_name,
            "n_trials": len(study.trials),
        },
        "best_trial": {
            "number": best_trial.number,
            "params": best_trial.params,
            "user_attrs": best_trial.user_attrs,
            "values": best_trial.values,
        },
        "args": get_vars(args),
        "slurm_stdout": slurm_stdout,
    }
    log_dir = Path("logs") / f"{args.cohort}-{args.task}-{args.method}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{log_name}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f)


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

    log_name = f"f{args.fold}-s{args.seed}-{args.suffix}"
    study = tune(
        train1fold,
        attrs,
        args,
        log_name=log_name,
    )
    if not args.tune:
        results = train1fold(attrs, args)
        pprint.pp(get_vars(args))
        write_summary(args, study, log_name)
