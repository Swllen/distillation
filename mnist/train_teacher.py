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
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from common import HERE, add_training_args, config_dict, load_mnist, make_model, parse_args, save_run, train


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_training_args(p, HERE / "configs" / "teacher.toml")
    args = parse_args(p)
    print(json.dumps(config_dict(args), indent=2))

    torch.manual_seed(args.seed)
    data = load_mnist(args.device)
    model = make_model(args)
    history = train(model, args, data, lambda logits, x, y: F.cross_entropy(logits, y))
    save_run(model, args, history)


if __name__ == "__main__":
    main()
