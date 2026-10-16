"""Train the MNIST teacher of Hinton, Vinyals & Dean (2015), Section 3.

Paper: "a single large neural net with two hidden layers of 1200 rectified linear
hidden units on all 60,000 training cases. The net was strongly regularized using
dropout and weight-constraints as described in [5] and the input images were
jittered by up to two pixels in any direction. This net achieved 67 test errors."

The paper gives no training hyperparameters, so they are taken from its reference [5],
Hinton et al. (2012), "Improving neural networks by preventing co-adaptation of
feature detectors", Appendix A (MNIST): https://arxiv.org/abs/1207.0580

  update  dw_t = p_t dw_{t-1} - (1 - p_t) eps_t <grad>,  w_t = w_{t-1} + dw_t
  with eps_t = lr * lr_decay^t and p_t rising linearly from momentum_initial to
  momentum_final over the first momentum_epochs epochs (t counts epochs from 0),
  then a max-norm projection of every hidden unit's incoming weight vector.

All settings live in a TOML file (default: configs/teacher.toml); see that file for
what each one means and where its value comes from. Any setting can also be given on
the command line, which takes precedence over the file:

  uv run mnist/train_teacher.py --config mnist/configs/teacher.toml --lr 1 --no-bias

Training metrics go to Weights & Biases (project distillation-mnist) unless --no-wandb.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from common import (
    HERE,
    add_training_args,
    args_from_dict,
    config_dict,
    load_mnist,
    make_model,
    parse_args,
    save_run,
    start_run,
    train,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_training_args(p, HERE / "configs" / "teacher.toml")
    return p


def make_args(settings: dict, name: str, **kwargs) -> argparse.Namespace:
    """Settings from a dict instead of a config file (used by reproduce.ipynb)."""
    return args_from_dict(build_parser(), settings, name, **kwargs)


def main() -> None:
    run_teacher(parse_args(build_parser()))


def run_teacher(args: argparse.Namespace) -> list[dict]:
    """Train the teacher and save it; returns the evaluation history."""
    print(json.dumps(config_dict(args), indent=2))
    run = start_run(args, job="teacher", paper_test_errors=67)
    torch.manual_seed(args.seed)
    data = load_mnist(args.device)
    model = make_model(args)
    history = train(model, args, data, lambda logits, x, y: {"loss": F.cross_entropy(logits, y)}, run)
    save_run(model, args, history, run)
    return history


if __name__ == "__main__":
    main()
