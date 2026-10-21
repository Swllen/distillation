"""Bar chart of every method's finished run for one seed against the paper's Table 1.

    uv run CRD_ICLR2020/scripts/plot_methods.py --seed 0 --out methods_seed0.png

reproduce.ipynb calls plot_methods() from here.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from run_experiment import ROOT, STAGES, read_metrics

# Table 1 of the paper (resnet32x4 -> resnet8x4); CRD+KD is not reported for this pair.
PAPER = {"teacher": 79.42, "vanilla": 72.50, "kd": 73.33, "crd": 75.51, "crd_kd": None,
         "fitnet": 73.50, "at": 73.44, "sp": 72.94, "cc": 72.97, "vid": 73.09, "rkd": 71.90,
         "pkt": 73.64, "ab": 73.17, "ft": 72.86, "fsp": 72.62, "nst": 73.30}
NAMES = {"vanilla": "vanilla", "kd": "KD", "crd": "CRD", "crd_kd": "CRD+KD", "fitnet": "FitNet", "at": "AT",
         "sp": "SP", "cc": "CC", "vid": "VID", "rkd": "RKD", "pkt": "PKT", "ab": "AB", "ft": "FT",
         "fsp": "FSP", "nst": "NST"}


def last_top1(runs_dir, seed):
    """{stage: last-epoch test top-1} of the completed, non-smoke runs of `seed` (trial 1 if there are several)."""
    found = {}
    for manifest_file in sorted(Path(runs_dir).glob("*/manifest.json")):
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest.get("smoke") or manifest.get("status") != "completed" or manifest.get("seed") != seed:
            continue
        stage = manifest["stage"]
        if stage not in found or manifest.get("trial") == 1:
            found[stage] = read_metrics(manifest_file.parent)[-1]["test_acc"]
    return {stage: found[stage] for stage in STAGES if stage in found}


def plot_methods(runs_dir=ROOT / "runs", seed=0, ax=None):
    """Ours (one seed, last epoch) next to the paper for every method, best at the top."""
    ours = last_top1(runs_dir, seed)
    if not ours:
        raise ValueError(f"no completed runs of seed {seed} in {runs_dir}")
    order = sorted(ours, key=ours.get)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.42 * len(order) + 1.3))
    y = range(len(order))
    ax.barh([i + 0.18 for i in y], [ours[s] for s in order], height=0.34, color="#2a78d6", zorder=3,
            label=f"Ours (seed {seed}, last epoch)")
    paper = [(i, PAPER[s]) for i, s in enumerate(order) if PAPER.get(s) is not None]
    ax.barh([i - 0.18 for i, _ in paper], [p for _, p in paper], height=0.34, color="#d3d1c7",
            edgecolor="#888780", hatch="////", linewidth=0, zorder=3, label="Paper (Table 1)")
    if "vanilla" in ours:
        ax.axvline(ours["vanilla"], color="#e34948", linestyle="--", linewidth=1.2, zorder=4,
                   label=f"vanilla {ours['vanilla']:.2f}")
    values = list(ours.values()) + [p for _, p in paper]
    low, high = int(min(values) - 0.5), max(values) + 0.5
    ax.set_xlim(low, high)
    ax.set_yticks(list(y), [NAMES.get(s, s) for s in order])
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:g}%")
    ax.set_xlabel("Test top-1 (last epoch)")
    ax.grid(axis="x", color="#e1e0d9", zorder=0)
    ax.tick_params(left=False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    first = [labels.index(label) for label in labels if not label.startswith("vanilla")]  # ours, paper, vanilla
    order_legend = first + [i for i in range(len(labels)) if i not in first]
    ax.legend([handles[i] for i in order_legend], [labels[i] for i in order_legend],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9)
    return ax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("methods.png"))
    args = parser.parse_args()
    ax = plot_methods(args.runs, args.seed)
    ax.figure.tight_layout()
    ax.figure.savefig(args.out, dpi=200)
    print("saved", args.out)


if __name__ == "__main__":
    main()
