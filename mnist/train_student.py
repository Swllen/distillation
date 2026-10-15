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
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from common import HERE, add_training_args, config_dict, load_mnist, load_model, make_model, parse_args, save_run, test_errors, train


def distillation_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, y: torch.Tensor, temperature: float, hard_weight: float
) -> torch.Tensor:
    t = temperature
    soft_targets = F.softmax(teacher_logits / t, dim=1)
    soft = -(soft_targets * F.log_softmax(student_logits / t, dim=1)).sum(dim=1).mean()
    hard = F.cross_entropy(student_logits, y)
    return (1 - hard_weight) * t * t * soft + hard_weight * hard


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_training_args(p, HERE / "configs" / "student_distill.toml")
    p.add_argument("--teacher", help='teacher checkpoint, relative paths are under mnist/; "" trains on hard labels only')
    p.add_argument("--temperature", type=float)
    p.add_argument("--hard-weight", type=float, help="weight of the hard-label term, 0..1")
    args = parse_args(p)
    print(json.dumps(config_dict(args), indent=2))

    torch.manual_seed(args.seed)
    data = load_mnist(args.device)
    model = make_model(args)

    if args.teacher:
        path = Path(args.teacher)
        teacher = load_model(path if path.is_absolute() else HERE / path, args.device)
        print(f"teacher {args.teacher}: {test_errors(teacher, data[2], data[3])} test errors")

        def loss_fn(logits: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                teacher_logits = teacher(x)
            return distillation_loss(logits, teacher_logits, y, args.temperature, args.hard_weight)
    else:

        def loss_fn(logits: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return F.cross_entropy(logits, y)

    history = train(model, args, data, loss_fn)
    save_run(model, args, history)


if __name__ == "__main__":
    main()
