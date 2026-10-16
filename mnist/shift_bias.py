"""Raise the output bias of digits a model never saw in training, and count test errors.

Paper, Section 3: with all 3s left out of the transfer set, the distilled net "makes
206 test errors of which 133 are on the 1010 threes in the test set. Most of the errors
are caused by the fact that the learned bias for the 3 class is much too low. If this
bias is increased by 3.5 (which optimizes overall performance on the test set), the
distilled model makes 109 errors of which 14 are on 3s."

Adding s to a class's output bias is the same as adding s to its logit, so this script
evaluates every shift from the logits of one forward pass. Like the paper, it picks the
shift on the test set, so the best row is an optimistic number.

  uv run mnist/shift_bias.py mnist/checkpoints/distill_no3.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import load_mnist, load_model, logits_of


def shift_errors(logits: torch.Tensor, y: torch.Tensor, digit: int, shifts: list[float]) -> list[tuple[float, int, int]]:
    """For each shift s added to `digit`'s logit (= its output bias): (s, errors, errors on `digit`)."""
    rows = []
    for s in shifts:
        z = logits.clone()
        z[:, digit] += s
        wrong = z.argmax(dim=1) != y
        rows.append((s, int(wrong.sum()), int((wrong & (y == digit)).sum())))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--digits", type=int, nargs="+", help="classes to shift; default the checkpoint's exclude_digits")
    p.add_argument("--max-shift", type=float, default=10.0)
    p.add_argument("--step", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    digits = args.digits or ckpt["config"].get("exclude_digits") or []
    if not digits:
        p.error("the checkpoint has no exclude_digits; pass --digits")

    model = load_model(args.checkpoint, args.device)
    _, _, x_test, y_test = load_mnist(args.device)
    logits = logits_of(model, x_test)
    shifts = torch.arange(0, args.max_shift + 1e-9, args.step).tolist()

    for d in digits:
        n_d = int((y_test == d).sum())
        print(f"\nshift added to the bias of class {d}  ({n_d} test images of {d})")
        print(f"{'shift':>6}  {'errors':>6}  {'on ' + str(d):>6}")
        rows = shift_errors(logits, y_test, d, shifts)
        best = min(rows, key=lambda r: r[1])
        for s, errors, on_d in rows:
            print(f"{s:6.1f}  {errors:6d}  {on_d:6d}{'   <- fewest errors' if (s, errors, on_d) == best else ''}")
        print(f"no shift: {rows[0][1]} errors ({rows[0][2]} on {d});  best shift {best[0]:.1f}: {best[1]} errors ({best[2]} on {d})")


if __name__ == "__main__":
    main()
