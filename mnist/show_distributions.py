"""Print the teacher's and a student's output distributions side by side.

For a few images it prints, per class, the teacher's and student's raw logits v and z,
the teacher's p = softmax(v / T), the student's q = softmax(z / T) and the student's
prediction at T = 1. Then, over the whole split, it summarizes the logit scales and how
far apart the distributions are: entropies, mean largest probability, how often the two
predict the same class, and KL in both directions.

The teacher checkpoint and T default to the ones the student was trained with.

  uv run mnist/show_distributions.py mnist/checkpoints/soft_swapped.pt
  uv run mnist/show_distributions.py mnist/checkpoints/soft_forward.pt --split test --n 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from common import HERE, load_mnist, load_model, logits_of


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("student", type=Path, help="student checkpoint")
    p.add_argument("--teacher", type=Path, help="teacher checkpoint; default the one the student was trained with")
    p.add_argument("--temperature", type=float, help="default the student's training temperature")
    p.add_argument("--split", choices=("train", "test"), default="train", help="train = the transfer set (default)")
    p.add_argument("--n", type=int, default=5, help="number of images to print")
    p.add_argument("--seed", type=int, default=0, help="which images to print")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = torch.load(args.student, map_location="cpu")["config"]
    teacher_path = args.teacher or (Path(cfg["teacher"]) if cfg.get("teacher") else None)
    if teacher_path is None:
        p.error("the student has no teacher in its config; pass --teacher")
    if not teacher_path.is_absolute() and not teacher_path.exists():
        teacher_path = HERE / teacher_path
    t = args.temperature or cfg.get("temperature") or 20.0

    x_train, y_train, x_test, y_test = load_mnist(args.device)
    x, y = (x_train, y_train) if args.split == "train" else (x_test, y_test)
    v = logits_of(load_model(teacher_path, args.device), x)
    z = logits_of(load_model(args.student, args.device), x)
    log_p, log_q = F.log_softmax(v / t, dim=1), F.log_softmax(z / t, dim=1)
    prob_p, prob_q, q1 = log_p.exp(), log_q.exp(), F.softmax(z, dim=1)

    print(f"student {args.student} (soft_loss {cfg.get('soft_loss', 'forward')}), teacher {teacher_path}, T = {t:g}, {args.split} set")
    picks = torch.randperm(len(x), generator=torch.Generator().manual_seed(args.seed))[: args.n].tolist()
    for i in picks:
        print(f"\nimage {i}: label {int(y[i])}, teacher predicts {int(v[i].argmax())}, student predicts {int(z[i].argmax())}")
        print(f"{'class':>5}  {'teacher logit v':>15}  {'student logit z':>15}  {'teacher p (T)':>13}  {'student q (T)':>13}  {'student (T=1)':>13}")
        for c in range(10):
            print(
                f"{c:5d}  {v[i, c].item():15.3f}  {z[i, c].item():15.3f}  "
                f"{prob_p[i, c].item():13.4f}  {prob_q[i, c].item():13.4f}  {q1[i, c].item():13.4f}"
            )

    def entropy(pr: torch.Tensor, lg: torch.Tensor) -> float:
        return -(pr * lg).sum(dim=1).mean().item()

    def spread(logits: torch.Tensor) -> float:
        return (logits.max(dim=1).values - logits.min(dim=1).values).mean().item()

    print(f"\nover all {len(x)} {args.split} images:")
    print(f"  logit range (max - min per image) teacher {spread(v):.3f}   student {spread(z):.3f}")
    print(f"  logit std (per image)             teacher {v.std(dim=1).mean():.3f}   student {z.std(dim=1).mean():.3f}")
    print(f"at T = {t:g}:")
    print(f"  entropy             teacher {entropy(prob_p, log_p):.4f}   student {entropy(prob_q, log_q):.4f}   (uniform: {torch.log(torch.tensor(10.0)):.4f})")
    print(f"  largest probability teacher {prob_p.max(dim=1).values.mean():.4f}   student {prob_q.max(dim=1).values.mean():.4f}")
    print(f"  same predicted class {(v.argmax(dim=1) == z.argmax(dim=1)).float().mean():.2%}")
    print(f"  KL(teacher || student) {(prob_p * (log_p - log_q)).sum(dim=1).mean():.5f}   KL(student || teacher) {(prob_q * (log_q - log_p)).sum(dim=1).mean():.5f}")


if __name__ == "__main__":
    main()
