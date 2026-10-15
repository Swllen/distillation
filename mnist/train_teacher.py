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
import time
import tomllib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import MLP, jitter, load_mnist, test_errors

HERE = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=HERE / "configs" / "teacher.toml")
    p.add_argument("--out", type=Path, help="checkpoint path; default checkpoints/<config name>.pt")
    p.add_argument("--device", help="default: cuda if available, else cpu")
    # Every setting below defaults to None and is filled in from the config file.
    p.add_argument("--hidden", type=int)
    p.add_argument("--bias", action=argparse.BooleanOptionalAction)
    p.add_argument("--p-in", type=float)
    p.add_argument("--p-hidden", type=float)
    p.add_argument("--max-norm", type=float)
    p.add_argument("--jitter", type=int)
    p.add_argument("--init-std", type=float)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--lr-decay", type=float)
    p.add_argument("--momentum-initial", type=float)
    p.add_argument("--momentum-final", type=float)
    p.add_argument("--momentum-epochs", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--eval-every", type=int)
    args = p.parse_args(argv)

    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    unknown = sorted(set(config) - set(vars(args)) - {"config"})
    if unknown:
        p.error(f"unknown setting(s) in {args.config}: {', '.join(unknown)}")
    for key, value in config.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    missing = [k for k, v in vars(args).items() if v is None and k not in ("out", "device")]
    if missing:
        p.error(f"missing setting(s), add them to {args.config}: {', '.join(missing)}")

    if args.out is None:
        args.out = HERE / "checkpoints" / f"{args.config.stem}.pt"
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def lr_at(args: argparse.Namespace, epoch: int) -> float:
    return args.lr * args.lr_decay**epoch


def momentum_at(args: argparse.Namespace, epoch: int) -> float:
    if epoch >= args.momentum_epochs:
        return args.momentum_final
    return args.momentum_initial + (epoch / args.momentum_epochs) * (args.momentum_final - args.momentum_initial)


def main() -> None:
    args = parse_args()
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    print(json.dumps(config, indent=2))

    torch.manual_seed(args.seed)
    x_train, y_train, x_test, y_test = load_mnist(args.device)
    n = len(x_train)
    steps_per_epoch = n // args.batch_size

    model = MLP(args.hidden, args.p_in, args.p_hidden, bias=args.bias).to(args.device)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=args.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    params = list(model.parameters())
    deltas = [torch.zeros_like(q) for q in params]

    print(f"device={args.device}  params={sum(q.numel() for q in params):,}  steps={args.epochs * steps_per_epoch:,}")
    history = []
    start = time.time()
    for epoch in range(args.epochs):
        eps, mom = lr_at(args, epoch), momentum_at(args, epoch)
        model.train()
        perm = torch.randperm(n, device=args.device)
        total_loss = 0.0
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch_size : (i + 1) * args.batch_size]
            loss = F.cross_entropy(model(jitter(x_train[idx], args.jitter)), y_train[idx])
            grads = torch.autograd.grad(loss, params)
            with torch.no_grad():
                torch._foreach_mul_(deltas, mom)
                torch._foreach_add_(deltas, grads, alpha=-(1 - mom) * eps)
                torch._foreach_add_(params, deltas)
            if args.max_norm > 0:
                model.apply_max_norm(args.max_norm)
            total_loss += loss.item()

        done = epoch + 1
        if done % args.eval_every == 0 or done == args.epochs:
            errors = test_errors(model, x_test, y_test)
            history.append({"epoch": done, "train_loss": total_loss / steps_per_epoch, "test_errors": errors})
            print(
                f"epoch {done:4d}  loss {total_loss / steps_per_epoch:.4f}  test errors {errors:4d}  "
                f"eps {eps:.4f}  momentum {mom:.3f}  {time.time() - start:.0f}s",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config, "history": history}, args.out)
    args.out.with_suffix(".json").write_text(json.dumps({"config": config, "history": history}, indent=2))
    print(f"final test errors: {history[-1]['test_errors']} / {len(x_test)}  ->  {args.out}")


if __name__ == "__main__":
    main()
