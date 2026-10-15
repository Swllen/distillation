"""Train the MNIST teacher of Hinton, Vinyals & Dean (2015), Section 3.

Paper: "a single large neural net with two hidden layers of 1200 rectified linear
hidden units on all 60,000 training cases. The net was strongly regularized using
dropout and weight-constraints as described in [5] and the input images were
jittered by up to two pixels in any direction. This net achieved 67 test errors."

The paper gives no training hyperparameters, so they are taken from its reference [5],
Hinton et al. (2012), "Improving neural networks by preventing co-adaptation of
feature detectors", Appendix A (MNIST): https://arxiv.org/abs/1207.0580

  * dropout 0.2 on the input pixels and 0.5 on every hidden unit
  * max-norm: each hidden unit's incoming weight vector has squared length <= 15
  * minibatch 100, cross-entropy averaged over the minibatch, 3000 epochs
  * update  dw_t = p_t dw_{t-1} - (1 - p_t) eps_t <grad>,  w_t = w_{t-1} + dw_t
    with eps_t = 10 * 0.998^t and p_t rising linearly from 0.5 to 0.99 over
    the first 500 epochs, then 0.99 (t counts epochs from 0)
  * weights initialized from N(0, 0.01^2); no weight decay

Not stated in either paper, chosen here: biases start at 0, and the jitter shift
(dy, dx) is drawn uniformly from {-2, ..., 2}^2 for every image at every epoch.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import MLP, jitter, load_mnist, test_errors

HERE = Path(__file__).resolve().parent


def lr_at(args: argparse.Namespace, epoch: int) -> float:
    return args.lr * args.lr_decay**epoch


def momentum_at(args: argparse.Namespace, epoch: int) -> float:
    if epoch >= args.momentum_epochs:
        return args.momentum_final
    return args.momentum_initial + (epoch / args.momentum_epochs) * (args.momentum_final - args.momentum_initial)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hidden", type=int, default=1200)
    p.add_argument("--p-in", type=float, default=0.2, help="dropout on the input pixels")
    p.add_argument("--p-hidden", type=float, default=0.5, help="dropout on the hidden units")
    p.add_argument("--max-norm", type=float, default=math.sqrt(15), help="max L2 norm of a hidden unit's incoming weights; 0 disables")
    p.add_argument("--jitter", type=int, default=2, help="max shift in pixels; 0 disables")
    p.add_argument("--init-std", type=float, default=0.01, help="std of the normal weight init")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--lr", type=float, default=10.0, help="eps_0 in the update rule above")
    p.add_argument("--lr-decay", type=float, default=0.998, help="eps is multiplied by this after every epoch")
    p.add_argument("--momentum-initial", type=float, default=0.5)
    p.add_argument("--momentum-final", type=float, default=0.99)
    p.add_argument("--momentum-epochs", type=int, default=500, help="epochs over which momentum ramps up linearly")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=25, help="epochs between test evaluations")
    p.add_argument("--out", type=Path, default=HERE / "checkpoints" / "teacher.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    x_train, y_train, x_test, y_test = load_mnist(args.device)
    n = len(x_train)
    steps_per_epoch = n // args.batch_size

    model = MLP(args.hidden, args.p_in, args.p_hidden).to(args.device)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=args.init_std)
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
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    torch.save({"state_dict": model.state_dict(), "config": config, "history": history}, args.out)
    args.out.with_suffix(".json").write_text(json.dumps({"config": config, "history": history}, indent=2))
    print(f"final test errors: {history[-1]['test_errors']} / {len(x_test)}  ->  {args.out}")


if __name__ == "__main__":
    main()
