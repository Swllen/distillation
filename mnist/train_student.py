"""Train the MNIST student nets of Hinton, Vinyals & Dean (2015), Section 3.

Paper: "A smaller net with two hidden layers of 800 rectified linear hidden units and
no regularization achieved 146 errors. But if the smaller net was regularized solely
by adding the additional task of matching the soft targets produced by the large net
at a temperature of 20, it achieved 74 test errors."

Two modes, chosen by the `teacher` setting:

  * teacher = ""        hard labels only:  loss = CE(y, softmax(z))
  * teacher = <ckpt>    distillation (paper Section 2):
        loss = (1 - hard_weight) * T^2 * CE(softmax(v / T), softmax(z / T))
             + hard_weight * CE(y, softmax(z))
    where z are the student logits and v the teacher logits on the same images.
    The T^2 factor keeps the soft-target gradients on the same scale as the
    hard-target ones when T changes (paper Section 2). The paper gives no weights
    for MNIST, only that the hard term should get "a considerably lower weight".

The student is evaluated at T = 1. The optimizer is the same as the teacher's (see
common.train); all settings live in configs/student_hard.toml and
configs/student_distill.toml, and flags override them:

  uv run mnist/train_student.py --config mnist/configs/student_distill.toml --temperature 8

Training metrics go to Weights & Biases (project distillation-mnist) unless --no-wandb.
For distillation runs the loss is split into soft_kl, teacher_entropy and hard_ce.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from common import (
    HERE,
    add_training_args,
    args_from_dict,
    config_dict,
    load_data,
    load_model,
    make_model,
    parse_args,
    save_run,
    start_run,
    test_errors,
    train,
)


def distillation_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, y: torch.Tensor, temperature: float, hard_weight: float
) -> dict[str, torch.Tensor]:
    """Loss terms; "loss" is minimized. soft_ce = teacher_entropy + soft_kl, all at temperature T and
    without the T^2 factor, so soft_kl is the part the student can reduce (its minimum is 0)."""
    t = temperature
    log_p = F.log_softmax(teacher_logits / t, dim=1)
    p = log_p.exp()
    soft = -(p * F.log_softmax(student_logits / t, dim=1)).sum(dim=1).mean()
    entropy = -(p * log_p).sum(dim=1).mean()
    hard = F.cross_entropy(student_logits, y)
    return {
        "loss": (1 - hard_weight) * t * t * soft + hard_weight * hard,
        "soft_ce": soft,
        "soft_kl": soft - entropy,
        "teacher_entropy": entropy,
        "hard_ce": hard,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_training_args(p, HERE / "configs" / "student_distill.toml")
    p.add_argument("--teacher", help='teacher checkpoint, relative paths are under mnist/; "" trains on hard labels only')
    p.add_argument("--temperature", type=float)
    p.add_argument("--hard-weight", type=float, help="weight of the hard-label term, 0..1")
    return p


def make_args(settings: dict, name: str, **kwargs) -> argparse.Namespace:
    """Settings from a dict instead of a config file (used by reproduce.ipynb)."""
    return args_from_dict(build_parser(), settings, name, **kwargs)


def main() -> None:
    run_student(parse_args(build_parser()))


def run_student(args: argparse.Namespace) -> list[dict]:
    """Train one student and save it; returns the evaluation history."""
    print(json.dumps(config_dict(args), indent=2))

    # Paper numbers for the 800-800 net: 146 test errors on hard labels; distilled at T = 20,
    # 74, or 206 when all 3s are left out of the transfer set.
    paper = None
    if not args.teacher and not args.exclude_digits:
        paper = 146
    elif args.teacher and args.temperature == 20:
        paper = {(): 74, (3,): 206}.get(tuple(args.exclude_digits))
    run = start_run(args, job="student", paper_test_errors=paper)
    torch.manual_seed(args.seed)
    data = load_data(args)
    model = make_model(args)

    if args.teacher:
        path = Path(args.teacher)
        teacher = load_model(path if path.is_absolute() else HERE / path, args.device)
        teacher_errors = test_errors(teacher, data[2], data[3])
        run.summary["teacher_test_errors"] = teacher_errors
        print(f"teacher {args.teacher}: {teacher_errors} test errors")

        def loss_fn(logits: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
            with torch.no_grad():
                teacher_logits = teacher(x)
            return distillation_loss(logits, teacher_logits, y, args.temperature, args.hard_weight)
    else:

        def loss_fn(logits: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"loss": F.cross_entropy(logits, y)}

    history = train(model, args, data, loss_fn, run)
    save_run(model, args, history, run)
    return history


if __name__ == "__main__":
    main()
