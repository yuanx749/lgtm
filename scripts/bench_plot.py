import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

AXIS_LABEL = {
    "samples": "Samples ($N$)",
    "features": "Taxa ($D$)",
    "subjects": "Subjects ($P$)",
    "components": "GP components ($R$)",
}


def load(in_dir):
    records = [json.loads(p.read_text()) for p in Path(in_dir).glob("bench-*.json")]
    by_axis = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_axis[r["axis"]][r["value"]].append(r)
    return by_axis


def plot_axis(axis, value_runs, out_dir):
    values = sorted(value_runs)
    median = np.array(
        [np.median([r["run_time"] for r in value_runs[v]]) for v in values]
    )
    slope = np.polyfit(np.log10(values), np.log10(median), 1)[0]

    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    for v in values:
        ax.scatter(
            [v] * len(value_runs[v]),
            [r["run_time"] for r in value_runs[v]],
            s=12,
            color="C0",
            alpha=0.35,
            zorder=1,
        )
    ax.plot(values, median, "-", color="C0", linewidth=1.5, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(values)
    ax.set_xticklabels([str(v) for v in values])
    ax.set_ylim(20, 500)
    ax.set_xlabel(AXIS_LABEL[axis])
    ax.set_ylabel("Training time (s)")
    fig.tight_layout()
    out_path = Path(out_dir) / f"bench_{axis}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return slope, values, median


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="output/bench")
    parser.add_argument("--out_dir", type=str, default="output/bench")
    a = parser.parse_args()

    by_axis = load(a.in_dir)
    all_records = [r for runs in by_axis.values() for rs in runs.values() for r in rs]
    svd = [r["svd_time"] for r in all_records]
    mem = [r["peak_rss_mb"] for r in all_records]

    for axis in AXIS_LABEL:
        slope, values, median = plot_axis(axis, by_axis[axis], a.out_dir)
        print(f"{axis}: slope={slope:.2f} "
              f"values={values} run_time={[round(m, 1) for m in median]}")
    print(f"svd_time s: min={min(svd):.4f} max={max(svd):.4f}")
    print(f"peak_rss MB: min={min(mem):.0f} max={max(mem):.0f}")
