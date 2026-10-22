"""Plots of what a model predicts for each true digit: a confusion matrix and a Sankey
(flow) diagram from true to predicted digits. Used by reproduce.ipynb.

  pred = predictions(model, x_test, shift_digits=[3], shift=2.5)
  show_predictions(y_test, pred, highlight=[3], title="no 3s, bias of 3 + 2.5")
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import logits_of


def predictions(model, x: torch.Tensor, shift_digits=(), shift: float = 0.0) -> torch.Tensor:
    """Predicted classes, after adding `shift` to the output biases (logits) of `shift_digits`."""
    z = logits_of(model, x)
    if len(shift_digits):
        z[:, list(shift_digits)] += shift
    return z.argmax(dim=1).cpu()


def confusion(y: torch.Tensor, pred: torch.Tensor) -> np.ndarray:
    """cm[i, j] = number of images of digit i predicted as j."""
    return np.bincount(10 * y.cpu().numpy() + pred.numpy(), minlength=100).reshape(10, 10)


def plot_confusion(cm: np.ndarray, ax, highlight=(), title: str = "") -> None:
    """Rows are true digits, columns predictions. The colour is the fraction of each row,
    the number is the count; the diagonal holds the correct predictions."""
    frac = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7, color="white" if frac[i, j] > 0.5 else "black")
    ax.set(xticks=range(10), yticks=range(10), xlabel="predicted digit", ylabel="true digit", title=title)
    for d in highlight:
        ax.add_patch(plt.Rectangle((-0.5, d - 0.5), 10, 1, fill=False, edgecolor="crimson", linewidth=2))
        ax.get_yticklabels()[d].set_color("crimson")


def plot_flows(cm: np.ndarray, ax, highlight=(), errors_only: bool = True, title: str = "") -> None:
    """Sankey diagram from true digits (left) to predicted digits (right); band widths are
    image counts. With errors_only, the correct predictions (the diagonal) are left out,
    since they would dwarf the mistakes."""
    flows = cm.copy()
    if errors_only:
        np.fill_diagonal(flows, 0)
    total = flows.sum()
    if total == 0:
        ax.text(0.5, 0.5, "no errors", ha="center", va="center")
        ax.axis("off")
        return
    gap, bar = 0.012, 0.04
    scale = (1 - 9 * gap) / total
    colors = plt.get_cmap("tab10")

    def stack(sizes):  # bottom of each node in a column, top to bottom = digits 0..9
        tops, y = [], 1.0
        for s in sizes:
            tops.append(y)
            y -= s * scale + gap
        return tops

    left_top, right_top = stack(flows.sum(axis=1)), stack(flows.sum(axis=0))
    left_used, right_used = np.zeros(10), np.zeros(10)
    t = np.linspace(0, 1, 50)
    ease = 3 * t**2 - 2 * t**3
    xs = bar + (1 - 2 * bar) * t
    for i in range(10):
        for j in range(10):
            n = flows[i, j]
            if not n:
                continue
            h = n * scale
            y0 = left_top[i] - left_used[i]
            y1 = right_top[j] - right_used[j]
            left_used[i] += h
            right_used[j] += h
            upper = y0 + (y1 - y0) * ease
            strong = not highlight or i in highlight
            ax.fill_between(xs, upper - h, upper, color=colors(i), alpha=0.6 if strong else 0.15, linewidth=0)
    for d in range(10):
        for x, top, size, side in ((0, left_top[d], flows[d].sum(), "true"), (1 - bar, right_top[d], flows[:, d].sum(), "pred")):
            if not size:
                continue
            ax.add_patch(plt.Rectangle((x, top - size * scale), bar, size * scale, color=colors(d)))
            ax.text(-0.01 if side == "true" else 1.01, top - size * scale / 2, f"{d} ({size})", va="center",
                    ha="right" if side == "true" else "left", fontsize=8,
                    color="crimson" if d in highlight and side == "true" else "black",
                    fontweight="bold" if d in highlight and side == "true" else "normal")
    ax.text(bar / 2, 1.03, "true digit", ha="center", fontsize=9)
    ax.text(1 - bar / 2, 1.03, "predicted as", ha="center", fontsize=9)
    ax.set(xlim=(-0.12, 1.12), ylim=(-0.02, 1.07), title=title + (" (misclassified images only)" if errors_only else ""))
    ax.axis("off")


def show_predictions(y: torch.Tensor, pred: torch.Tensor, highlight=(), title: str = "") -> None:
    """Confusion matrix and Sankey diagram side by side."""
    cm = confusion(y, pred)
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1.2]})
    plot_confusion(cm, a, highlight, f"{title}: {cm.sum() - np.trace(cm)} errors")
    plot_flows(cm, b, highlight, title="where the errors go")
    plt.tight_layout()
    plt.show()
