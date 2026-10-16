"""Raise the output bias of digits a model never saw in training, and count test errors.

Paper, Section 3: with all 3s left out of the transfer set, the distilled net "makes
206 test errors of which 133 are on the 1010 threes in the test set. Most of the errors
are caused by the fact that the learned bias for the 3 class is much too low. If this
bias is increased by 3.5 (which optimizes overall performance on the test set), the
distilled model makes 109 errors of which 14 are on 3s." With a transfer set of only
7s and 8s, it makes 47.3% test errors, and 13.2% after the biases of 7 and 8 are
reduced by 7.6.

All digits in --digits (default: the checkpoint's exclude_digits) get the same shift.
Adding s to the unseen classes picks the same predictions as subtracting s from the
seen ones, so for the 7-and-8 model this sweep is the paper's bias reduction.

Adding s to a class's output bias is the same as adding s to its logit, so every shift
is evaluated from the logits of one forward pass. Like the paper, the script picks the
shift on the test set, so the best row is an optimistic number.

  uv run mnist/shift_bias.py mnist/checkpoints/distill_no3.pt
  uv run mnist/shift_bias.py mnist/checkpoints/distill_only78.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import load_mnist, load_model, logits_of


def shift_errors(
    logits: torch.Tensor, y: torch.Tensor, digits: int | list[int], shifts: list[float]
) -> list[tuple[float, int, int]]:
    """For each shift s added to the logits (= output biases) of `digits`:
    (s, errors, errors on test images of `digits`)."""
    digits = [digits] if isinstance(digits, int) else list(digits)
    of_digits = torch.isin(y, torch.tensor(digits, device=y.device))
    rows = []
    for s in shifts:
        z = logits.clone()
        z[:, digits] += s
        wrong = z.argmax(dim=1) != y
        rows.append((s, int(wrong.sum()), int((wrong & of_digits).sum())))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--digits", type=int, nargs="+", help="classes shifted together; default the checkpoint's exclude_digits")
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
    rows = shift_errors(logits_of(model, x_test), y_test, digits, torch.arange(0, args.max_shift + 1e-9, args.step).tolist())
    best = min(rows, key=lambda r: r[1])

    n = len(y_test)
    n_digits = int(torch.isin(y_test, torch.tensor(digits, device=y_test.device)).sum())
    print(f"\nshift added to the biases of {digits}  ({n_digits} of the {n} test images are of these digits)")
    print(f"{'shift':>6}  {'errors':>6}  {'rate':>6}  {'on them':>7}")
    for s, errors, on_d in rows:
        print(f"{s:6.1f}  {errors:6d}  {errors / n:6.1%}  {on_d:7d}{'   <- fewest errors' if (s, errors, on_d) == best else ''}")
    print(
        f"no shift: {rows[0][1]} errors ({rows[0][1] / n:.1%}), {rows[0][2]} on {digits};  "
        f"best shift {best[0]:.1f}: {best[1]} errors ({best[1] / n:.1%}), {best[2]} on {digits}"
    )


if __name__ == "__main__":
    main()
