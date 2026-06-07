import argparse
import importlib
import json
import os
import pprint
import sys
import time
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lgtm.utils import get_vars


def objective(train_func, attrs, args, trial: optuna.trial.Trial):
    if args.method == "dgbfgp":
        args.lr = trial.suggest_categorical("lr", [5e-4, 1e-3, 5e-3, 1e-2])
    else:
        args.lr = trial.suggest_categorical("lr", [5e-3, 1e-2, 5e-2])
    args.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    if args.encode_y or not args.linear_decoded:
        args.hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    print(os.environ.get("SLURM_PROCID"))
    print(f"Trial {trial.number}")
    results = train_func(attrs, args)
    print(os.environ.get("SLURM_PROCID"))
    print(f"Trial {trial.number}")
    print(trial.params)
    pprint.pp(results["metrics"])
    pprint.pp(results["meta"])
    metrics = results.get("metrics", {})
    meta = results.get("meta", {})
    test_metrics = metrics.get("test", {})
    recon_metrics = metrics.get("recon", {})
    attrs_to_set = {
        "best_epoch": meta.get("best_epoch"),
        "run_time": meta.get("run_time"),
        "test_mse": test_metrics.get("mse"),
        "test_cce": test_metrics.get("cce"),
        "jsd": test_metrics.get("jsd"),
        "bcd": test_metrics.get("bcd"),
        "recon_tcce": recon_metrics.get("tcce"),
    }
    for key, value in attrs_to_set.items():
        if value is not None:
            trial.set_user_attr(key, value)
    best_score = meta.get("best_score")
    return best_score


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
        args.save_pred = False
        args.save_model = False
        directions = ["minimize", "minimize"]
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
        best_trial = (
            min(study.best_trials, key=lambda t: t.values)
            if len(study.directions) > 1
            else study.best_trial
        )
        vars(args).update(best_trial.params)
    return study


def write_summary(args, results, study, log_name):
    best_trial = (
        min(study.best_trials, key=lambda t: t.values)
        if len(study.directions) > 1
        else study.best_trial
    )
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
        "metrics": results.get("metrics", {}),
        "meta": results.get("meta", {}),
        "slurm_stdout": slurm_stdout,
    }
    log_dir = Path("logs") / f"{args.cohort}-{args.task}-{args.method}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{log_name}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f)


if __name__ == "__main__":

    from lgtm.train import train1, train1fold, train5folds

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
            pprint.pp(results["metrics"])
            pprint.pp(results["meta"])
            write_summary(args, results, study, log_name)
    else:
        args.name += f"-L{args.latent_dim}-s{args.seed}"
        log_name = f"L{args.latent_dim}-s{args.seed}-{args.suffix}"
        study = tune(
            train5folds,
            attrs,
            args,
            log_name=log_name,
        )
        if not args.tune:
            results = train5folds(attrs, args)
            pprint.pp(get_vars(args))
            pprint.pp(results["metrics"])
            pprint.pp(results["meta"])
            write_summary(args, results, study, log_name)
