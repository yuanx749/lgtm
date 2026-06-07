import argparse
import copy
import json
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from lgtm.config import args as base_args
from lgtm.data import MetagenomeDataset
from lgtm.train import train_val_test
from lgtm.utils import nndsvd_init

from bench_data import AXES, make_attrs, resolve

REPS = 3
SEED = 42
CONFIGS = [
    (axis, value, rep)
    for axis in AXES
    for value in AXES[axis]
    for rep in range(REPS)
]


def build_args(latent_dim, n_epoch):
    args = copy.deepcopy(base_args)
    args.M = 5
    args.latent_dim = latent_dim
    args.hidden_dim = 64
    args.n_epoch = n_epoch
    args.patience = 0
    args.k = 1
    args.b = 0.001
    args.lr = 0.05
    args.batch_size = 64
    args.seed = SEED
    args.init_b = True
    args.encode_y = True
    args.normalize_latent = "mean"
    args.linear_decoded = True
    args.non_negative = True
    args.normalize_weight = True
    args.method = "gp"
    args.task = "dr"
    return args


def run(axis, value, rep, latent_dim, n_epoch, out_dir):
    n_samples, n_features, n_subjects, n_components = resolve(axis, value)
    attrs = make_attrs(n_samples, n_features, n_subjects, n_components, seed=SEED)
    args = build_args(latent_dim, n_epoch)

    t0 = time.perf_counter()
    nndsvd_init(attrs.y, latent_dim, random_state=SEED)
    svd_time = time.perf_counter() - t0

    dataset = MetagenomeDataset(attrs.y, attrs.x_num, attrs.x_cat, transform=None)
    results = train_val_test(attrs, args, dataset)
    run_time = results["meta"]["run_time"]

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    record = {
        "axis": axis,
        "value": value,
        "rep": rep,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_subjects": n_subjects,
        "n_components": n_components,
        "latent_dim": latent_dim,
        "n_epoch": n_epoch,
        "run_time": run_time,
        "per_epoch": run_time / n_epoch,
        "svd_time": svd_time,
        "peak_rss_mb": peak_rss_mb,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bench-{axis}-{value}-r{rep}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f)
    print(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--axis", type=str, default=None)
    parser.add_argument("--value", type=int, default=None)
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--latent_dim", type=int, default=5)
    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--out", type=str, default="output/bench")
    a = parser.parse_args()

    if a.index is not None:
        axis, value, rep = CONFIGS[a.index]
    else:
        axis, value, rep = a.axis, a.value, a.rep

    run(axis, value, rep, a.latent_dim, a.n_epoch, Path(a.out))
