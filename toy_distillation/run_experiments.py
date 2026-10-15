"""Reproduce the figures of "A toy model of distillation" (Ziming Liu, 2026).

Experiments
  temp    : Adam, T in {1, 0.1, 0.01}, 1000 steps, regular vs distill   (blog fig. 1)
  speedup : Adam, T = 0.1, 3000 steps, log-x, raw and aligned by ||p||_2 (blog fig. 2)
  sgd     : SGD,  T in {1, 0.1, 0.01}, 3000 steps                         (blog fig. 3)
  sweep   : Adam, many T; measured burn-in speedup vs predicted 1/||p||_2 ("Verify")
  batch   : the sweep repeated for batch sizes 1, 8, 32, 128, 512

Usage (from the repository root)
  uv run toy_distillation/run_experiments.py all
  uv run toy_distillation/run_experiments.py temp speedup --batch-size 64 --lr 1e-2
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toy_distill import Config, train

C_REG, C_DIS, C_ENT = "tab:blue", "tab:orange", "red"


def run_pair(base: Config, **kw) -> tuple[dict, dict]:
    cfg = replace(base, **kw)
    reg = train(replace(cfg, method="regular"))
    dis = train(replace(cfg, method="distill"))
    print(
        f"[{cfg.optimizer} T={cfg.temperature:g}] H={dis['entropy']:.4f} "
        f"||p||={dis['p_norm']:.4f} pred speedup={dis['speedup_pred']:.2f} | "
        f"final eval: regular={reg['eval_loss'][-1]:.4f} distill={dis['eval_loss'][-1]:.4f}"
    )
    return reg, dis


def plot_pair(ax, reg, dis, title, logx=False, align=False, use_eval=False):
    if use_eval:
        sr, yr = reg["eval_steps"], reg["eval_loss"]
        sd, yd = dis["eval_steps"], dis["eval_loss"]
    else:
        sr = sd = list(range(1, len(reg["train_loss"]) + 1))
        yr, yd = reg["train_loss"], dis["train_loss"]
    if align:
        sd = [s / dis["p_norm"] for s in sd]
    ax.plot(sr, yr, color=C_REG, lw=1, label="Regular (CE)")
    ax.plot(sd, yd, color=C_DIS, lw=1, label="Distillation (KL)")
    ax.axhline(dis["entropy"], color=C_ENT, ls="--", lw=1, label="Entropy (teacher)")
    ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel("Step (Regular) or Step / ||p||_2 (Distillation)" if align else "Step")
    ax.set_ylabel("Loss" + (" (eval)" if use_eval else ""))
    ax.set_title(title)
    ax.legend(fontsize=8)


def save(fig, out, name, payload):
    fig.tight_layout()
    path = os.path.join(out, name)
    fig.savefig(path + ".png", dpi=150)
    plt.close(fig)
    with open(path + ".json", "w") as f:
        json.dump(payload, f)
    print(f"saved {path}.png")


def exp_temp(base: Config, out: str, optimizer="adam", steps=1000, name="temp"):
    temps = [1.0, 0.1, 0.01]
    runs = {T: run_pair(base, optimizer=optimizer, steps=steps, temperature=T) for T in temps}
    for use_eval in (False, True):
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        for ax, T in zip(axes, temps):
            plot_pair(ax, *runs[T], f"Teacher temperature = {T:.3f}", use_eval=use_eval)
        fig.suptitle(f"{optimizer.upper()}, lr={base.lr:g}, batch={base.batch_size}")
        save(fig, out, name + ("_eval" if use_eval else ""), {str(T): r for T, r in runs.items()})


def exp_speedup(base: Config, out: str, T=0.1, steps=3000):
    reg, dis = run_pair(base, optimizer="adam", steps=steps, temperature=T)
    for use_eval in (False, True):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_pair(axes[0], reg, dis, f"Teacher temperature = {T:.3f}", logx=True, use_eval=use_eval)
        plot_pair(axes[1], reg, dis, f"Teacher temperature = {T:.3f}, aligned",
                  logx=True, align=True, use_eval=use_eval)
        fig.suptitle(f"||p||_2 = {dis['p_norm']:.3f}, predicted speedup 1/||p||_2 = {dis['speedup_pred']:.2f}")
        save(fig, out, "speedup" + ("_eval" if use_eval else ""), {"regular": reg, "distill": dis})


def steps_to_reach(steps, losses, target):
    for s, l in zip(steps, losses):
        if l <= target:
            return s
    return math.nan


SWEEP_TEMPS = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01]


def measure_speedups(base: Config, temps=SWEEP_TEMPS, steps=3000, frac=0.5) -> list[dict]:
    """Burn-in speedup: steps for the clean eval loss to cover `frac` of the gap
    between its initial value and the teacher entropy, regular / distill."""
    rows = []
    for T in temps:
        reg, dis = run_pair(base, optimizer="adam", steps=steps, temperature=T, eval_every=1)
        L0, H = dis["eval_loss"][0], dis["entropy"]
        target = H + (1 - frac) * (L0 - H)
        s_reg = steps_to_reach(reg["eval_steps"], reg["eval_loss"], target)
        s_dis = steps_to_reach(dis["eval_steps"], dis["eval_loss"], target)
        rows.append({"T": T, "batch_size": base.batch_size, "p_norm": dis["p_norm"],
                     "speedup_pred": dis["speedup_pred"],
                     "steps_regular": s_reg, "steps_distill": s_dis,
                     "speedup_measured": s_reg / s_dis if s_dis else math.nan})
        print(f"  B={base.batch_size} T={T:g}: steps regular={s_reg} distill={s_dis} "
              f"measured={rows[-1]['speedup_measured']:.2f} pred={dis['speedup_pred']:.2f}")
    return rows


def exp_sweep(base: Config, out: str, steps=3000, frac=0.5):
    rows = measure_speedups(base, steps=steps, frac=frac)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    pred = [r["speedup_pred"] for r in rows]
    meas = [r["speedup_measured"] for r in rows]
    Ts = [r["T"] for r in rows]
    axes[0].plot(Ts, pred, "o-", label="predicted 1/||p||_2")
    axes[0].plot(Ts, meas, "s-", label="measured")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("Teacher temperature T"); axes[0].set_ylabel("Speedup")
    axes[0].legend()
    lo, hi = min(pred + [m for m in meas if m == m]), max(pred + [m for m in meas if m == m])
    axes[1].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[1].scatter(pred, meas)
    for T, x, y in zip(Ts, pred, meas):
        axes[1].annotate(f"T={T:g}", (x, y), fontsize=7)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("predicted 1/||p||_2"); axes[1].set_ylabel("measured speedup")
    fig.suptitle(f"Burn-in speedup (Adam, {int(frac * 100)}% of gap closed)")
    save(fig, out, "sweep", rows)


def exp_batch(base: Config, out: str, batch_sizes=(1, 8, 32, 128, 512), steps=3000, frac=0.5):
    """Repeat the speedup sweep for several batch sizes. The blog's 1/||p||_2 argument
    is per-sample; larger batches average away label noise in the regular gradient."""
    rows = []
    for B in batch_sizes:
        rows += measure_speedups(replace(base, batch_size=B), steps=steps, frac=frac)

    colors = plt.cm.viridis([i / max(len(batch_sizes) - 1, 1) for i in range(len(batch_sizes))])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    first = [r for r in rows if r["batch_size"] == batch_sizes[0]]
    axes[0].plot([r["T"] for r in first], [r["speedup_pred"] for r in first],
                 "k--", lw=1.5, label="predicted 1/||p||_2")
    finite = [r["speedup_pred"] for r in rows] + [r["speedup_measured"] for r in rows
                                                 if r["speedup_measured"] == r["speedup_measured"]]
    lo, hi = min(finite), max(finite)
    axes[1].plot([lo, hi], [lo, hi], "k--", lw=1)
    for B, c in zip(batch_sizes, colors):
        rs = [r for r in rows if r["batch_size"] == B]
        axes[0].plot([r["T"] for r in rs], [r["speedup_measured"] for r in rs], "o-", color=c, label=f"B={B}")
        axes[1].plot([r["speedup_pred"] for r in rs], [r["speedup_measured"] for r in rs],
                     "o-", color=c, label=f"B={B}")
    for ax in axes:
        ax.set_xscale("log"); ax.set_yscale("log"); ax.legend(fontsize=8)
    axes[0].set_xlabel("Teacher temperature T"); axes[0].set_ylabel("Speedup")
    axes[1].set_xlabel("predicted 1/||p||_2"); axes[1].set_ylabel("measured speedup")
    fig.suptitle(f"Burn-in speedup vs batch size (Adam, lr={base.lr:g}, {int(frac * 100)}% of gap closed)")
    save(fig, out, "batch_sweep", rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="+", choices=["temp", "speedup", "sgd", "sweep", "batch", "all"])
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--sgd-lr", type=float, default=None,
                    help="lr for the SGD experiment (default: same as --lr, i.e. 1e-2). With 1e-2 SGD "
                         "descends much more slowly than in the blog's figure; try 0.1 or larger to match it.")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128, 512],
                    help="batch sizes for the 'batch' experiment")
    ap.add_argument("--vocab", type=int, default=1000)
    ap.add_argument("--d-in", type=int, default=10)
    ap.add_argument("--teacher-seed", type=int, default=0)
    ap.add_argument("--student-seed", type=int, default=1)
    ap.add_argument("--data-seed", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
                    help="output directory (default: results/ next to this script)")
    args = ap.parse_args()

    base = Config(lr=args.lr, batch_size=args.batch_size, vocab=args.vocab, d_in=args.d_in,
                  teacher_seed=args.teacher_seed, student_seed=args.student_seed,
                  data_seed=args.data_seed)
    if args.device:
        base = replace(base, device=args.device)
    os.makedirs(args.out, exist_ok=True)
    print(f"device={base.device}")

    exps = {"temp", "speedup", "sgd", "sweep", "batch"} if "all" in args.experiments else set(args.experiments)
    if "temp" in exps:
        exp_temp(base, args.out)
    if "speedup" in exps:
        exp_speedup(base, args.out)
    if "sgd" in exps:
        exp_temp(replace(base, lr=args.sgd_lr or args.lr), args.out, optimizer="sgd", steps=3000, name="sgd_temp")
    if "sweep" in exps:
        exp_sweep(base, args.out)
    if "batch" in exps:
        exp_batch(base, args.out, batch_sizes=tuple(args.batch_sizes))


if __name__ == "__main__":
    main()
