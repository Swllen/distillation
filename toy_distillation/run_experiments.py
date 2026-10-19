"""Reproduce the figures of "A toy model of distillation" (Ziming Liu, 2026).

Experiments
  temp    : Adam, T in {1, 0.1, 0.01}, 1000 steps, regular vs distill   (blog fig. 1)
  speedup : Adam, T = 0.1, 3000 steps, log-x, raw and aligned by ||p||_2 (blog fig. 2)
  sgd     : SGD,  T in {1, 0.1, 0.01}, 3000 steps                         (blog fig. 3)
  sweep   : Adam, many T; measured burn-in speedup vs predicted 1/||p||_2 ("Verify")
  batch   : the sweep repeated for batch sizes 1, 8, 32, 128, 512
  optim   : the sweep repeated for SGD, momentum, RMSprop, Adagrad, Adam, Adamax, signSGD
  losses  : Adam, T in {1, 0.1, 0.01}: hard label (regular) vs swapped CE -sum q log p

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


def measure_speedups(base: Config, temps=SWEEP_TEMPS, steps=3000, frac=0.5, keep_T=None):
    """Burn-in speedup with optimizer `base.optimizer`: steps for the clean eval loss to
    cover `frac` of the gap between its initial value and the teacher entropy, regular / distill.

    Returns the rows, or (rows, (regular, distill)) for temperature `keep_T` if it is given."""
    rows, kept = [], None
    for T in temps:
        reg, dis = run_pair(base, steps=steps, temperature=T, eval_every=1)
        if T == keep_T:
            kept = (reg, dis)
        L0, H = dis["eval_loss"][0], dis["entropy"]
        target = H + (1 - frac) * (L0 - H)
        s_reg = steps_to_reach(reg["eval_steps"], reg["eval_loss"], target)
        s_dis = steps_to_reach(dis["eval_steps"], dis["eval_loss"], target)
        rows.append({"T": T, "optimizer": base.optimizer, "lr": base.lr, "batch_size": base.batch_size,
                     "p_norm": dis["p_norm"], "speedup_pred": dis["speedup_pred"],
                     "steps_regular": s_reg, "steps_distill": s_dis,
                     "speedup_measured": s_reg / s_dis if s_dis else math.nan})
        print(f"  {base.optimizer} B={base.batch_size} T={T:g}: steps regular={s_reg} distill={s_dis} "
              f"measured={rows[-1]['speedup_measured']:.2f} pred={dis['speedup_pred']:.2f}")
    return rows if keep_T is None else (rows, kept)


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


LOSS_STYLES = {
    "regular": ("tab:blue", "Regular (sampled one-hot)"),
    "distill": ("tab:orange", "Distill: CE(p, q)"),
    "swapped": ("tab:green", "Swapped: -Σ q log p"),
    "reverse_kl": ("tab:purple", "Reverse KL(q || p)"),
    "teacher_argmax": ("tab:brown", "Teacher argmax one-hot"),
}


def exp_losses(base: Config, out: str, methods=("regular", "swapped"), temps=(1.0, 0.1, 0.01), steps=3000):
    """Does the swapped CE behave like one-hot (hard label) training?
    Rows: forward CE to the teacher, argmax agreement with the teacher, student entropy H(q)."""
    runs = {}
    for T in temps:
        for m in methods:
            r = train(replace(base, temperature=T, method=m, steps=steps))
            runs[(T, m)] = r
            print(f"[losses T={T:g} {m}] final eval CE={r['eval_loss'][-1]:.4f} "
                  f"(H={r['entropy']:.4f}) agree={r['eval_acc'][-1]:.3f} H(q)={r['eval_entropy'][-1]:.3f}")

    rows = [("eval_loss", "CE(p, q) (eval)", True), ("eval_acc", "argmax agreement", False),
            ("eval_entropy", "student entropy H(q)", False)]
    fig, axes = plt.subplots(len(rows), len(temps), figsize=(6 * len(temps), 3.6 * len(rows)), squeeze=False)
    for j, T in enumerate(temps):
        H = runs[(T, methods[0])]["entropy"]
        for i, (key, ylabel, logy) in enumerate(rows):
            ax = axes[i][j]
            for m in methods:
                color, label = LOSS_STYLES[m]
                r = runs[(T, m)]
                ax.plot(r["eval_steps"][1:], r[key][1:], color=color, lw=1.2, label=label)
            if key in ("eval_loss", "eval_entropy"):
                ax.axhline(H, color=C_ENT, ls="--", lw=1, label="Entropy (teacher)")
            ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel("Step"); ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(f"Teacher temperature = {T:g}")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"Hard label vs swapped CE (Adam, lr={base.lr:g}, batch={base.batch_size})")
    save(fig, out, "losses", {f"{T:g}/{m}": r for (T, m), r in runs.items()})


# Default lr per optimizer, chosen so that each one makes visible progress within 3000 steps.
# Normalised optimizers move ~lr per coordinate per step; plain SGD moves ~lr * |grad| with
# |grad| ~ 1/V per row, so it needs a much larger lr. Adagrad's step decays like lr / sqrt(t).
OPTIM_LRS = {"sgd": 1.0, "momentum": 0.1, "rmsprop": 1e-2, "adagrad": 1e-1,
             "adam": 1e-2, "adamax": 1e-2, "signsgd": 1e-2}


def parse_optimizers(specs: list[str]) -> list[tuple[str, float]]:
    """'adam' -> ('adam', OPTIM_LRS['adam']); 'rmsprop:3e-3' -> ('rmsprop', 3e-3)."""
    out = []
    for spec in specs:
        name, _, lr = spec.partition(":")
        if name not in OPTIM_LRS:
            raise SystemExit(f"unknown optimizer '{name}'; choose from {list(OPTIM_LRS)}")
        out.append((name, float(lr) if lr else OPTIM_LRS[name]))
    return out


def exp_optim(base: Config, out: str, optimizers, steps=3000, frac=0.5, curve_T=0.1):
    """Speedup of distillation over regular training for each optimizer, across temperatures,
    plus the eval-loss curves of every optimizer at T = curve_T."""
    rows, curves = [], {}
    for name, lr in optimizers:
        r, (reg, dis) = measure_speedups(replace(base, optimizer=name, lr=lr), steps=steps,
                                         frac=frac, keep_T=curve_T)
        rows += r
        curves[name] = {"lr": lr, "regular": reg, "distill": dis}

    colors = dict(zip([n for n, _ in optimizers], plt.cm.tab10.colors))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    first = [r for r in rows if r["optimizer"] == optimizers[0][0]]
    axes[0].plot([r["T"] for r in first], [r["speedup_pred"] for r in first],
                 "k--", lw=1.5, label="predicted 1/||p||_2")
    finite = [r["speedup_pred"] for r in rows] + [r["speedup_measured"] for r in rows
                                                 if r["speedup_measured"] == r["speedup_measured"]]
    lo, hi = min(finite), max(finite)
    axes[1].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[1].axhline(1, color="gray", lw=0.8, ls=":")
    for name, lr in optimizers:
        rs = [r for r in rows if r["optimizer"] == name]
        label = f"{name} (lr={lr:g})"
        axes[0].plot([r["T"] for r in rs], [r["speedup_measured"] for r in rs], "o-",
                     color=colors[name], label=label)
        axes[1].plot([r["speedup_pred"] for r in rs], [r["speedup_measured"] for r in rs], "o-",
                     color=colors[name], label=label)
    for ax in axes:
        ax.set_xscale("log"); ax.set_yscale("log"); ax.legend(fontsize=7)
    axes[0].set_xlabel("Teacher temperature T"); axes[0].set_ylabel("Speedup (regular steps / distill steps)")
    axes[1].set_xlabel("predicted 1/||p||_2"); axes[1].set_ylabel("measured speedup")
    fig.suptitle(f"Burn-in speedup by optimizer (batch={base.batch_size}, {int(frac * 100)}% of gap closed; "
                 f"NaN = not reached in {steps} steps)")
    save(fig, out, "optim_sweep", rows)

    ncols = 4
    nrows = math.ceil(len(optimizers) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.6 * nrows), squeeze=False)
    for ax, (name, lr) in zip(axes.flat, optimizers):
        c = curves[name]
        plot_pair(ax, c["regular"], c["distill"], f"{name}, lr={lr:g}", logx=True, use_eval=True)
    for ax in list(axes.flat)[len(optimizers):]:
        ax.axis("off")
    fig.suptitle(f"Eval loss at T = {curve_T:g} by optimizer (batch={base.batch_size})")
    save(fig, out, "optim_curves", curves)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="+", choices=["temp", "speedup", "sgd", "sweep", "batch", "optim", "losses", "all"])
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--sgd-lr", type=float, default=None,
                    help="lr for the SGD experiment (default: same as --lr, i.e. 1e-2). With 1e-2 SGD "
                         "descends much more slowly than in the blog's figure; try 0.1 or larger to match it.")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128, 512],
                    help="batch sizes for the 'batch' experiment")
    ap.add_argument("--optimizers", nargs="+", default=list(OPTIM_LRS),
                    help="optimizers for the 'optim' experiment, each 'name' or 'name:lr' "
                         f"(default lrs: {', '.join(f'{k}={v:g}' for k, v in OPTIM_LRS.items())})")
    ap.add_argument("--loss-methods", nargs="+", default=["regular", "swapped"], choices=list(LOSS_STYLES),
                    help="training methods for the 'losses' experiment (default: hard label vs swapped CE)")
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

    optimizers = parse_optimizers(args.optimizers)
    exps = ({"temp", "speedup", "sgd", "sweep", "batch", "optim", "losses"} if "all" in args.experiments
            else set(args.experiments))
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
    if "optim" in exps:
        exp_optim(base, args.out, optimizers)
    if "losses" in exps:
        exp_losses(base, args.out, methods=tuple(args.loss_methods))


if __name__ == "__main__":
    main()
