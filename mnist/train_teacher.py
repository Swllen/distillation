"""Train the MNIST teacher of Hinton, Vinyals & Dean (2015), Section 3.

Paper: "a single large neural net with two hidden layers of 1200 rectified linear
hidden units on all 60,000 training cases. The net was strongly regularized using
dropout and weight-constraints ... The input images were jittered by up to two
pixels in any direction. This net achieved 67 test errors."

The paper does not give the optimizer settings. The defaults here follow the dropout
recipe of Srivastava et al. (2014): dropout 0.2 on the input and 0.5 on the hidden
layers, a max-norm constraint on each hidden unit's incoming weights, SGD with
momentum and a large learning rate, decayed over training.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from common import MLP, jitter, load_mnist, test_errors

HERE = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hidden", type=int, default=1200)
    p.add_argument("--p-in", type=float, default=0.2, help="dropout on the input pixels")
    p.add_argument("--p-hidden", type=float, default=0.5, help="dropout on the hidden units")
    p.add_argument("--max-norm", type=float, default=3.5, help="max L2 norm of a hidden unit's incoming weights; 0 disables")
    p.add_argument("--jitter", type=int, default=2, help="max shift in pixels; 0 disables")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.1, help="peak learning rate, cosine-decayed to 0")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=5, help="epochs between test evaluations")
    p.add_argument("--out", type=Path, default=HERE / "checkpoints" / "teacher.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    x_train, y_train, x_test, y_test = load_mnist(args.device)
    n = len(x_train)
    steps_per_epoch = n // args.batch_size
    total_steps = args.epochs * steps_per_epoch

    model = MLP(args.hidden, args.p_in, args.p_hidden).to(args.device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total_steps)))

    print(f"device={args.device}  params={sum(q.numel() for q in model.parameters()):,}  steps={total_steps:,}")
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=args.device)
        total_loss = 0.0
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch_size : (i + 1) * args.batch_size]
            loss = F.cross_entropy(model(jitter(x_train[idx], args.jitter)), y_train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if args.max_norm > 0:
                model.apply_max_norm(args.max_norm)
            total_loss += loss.item()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            errors = test_errors(model, x_test, y_test)
            history.append({"epoch": epoch, "train_loss": total_loss / steps_per_epoch, "test_errors": errors})
            print(
                f"epoch {epoch:4d}  loss {total_loss / steps_per_epoch:.4f}  "
                f"test errors {errors:4d}  lr {sched.get_last_lr()[0]:.4f}  {time.time() - start:.0f}s",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    torch.save({"state_dict": model.state_dict(), "config": config, "history": history}, args.out)
    args.out.with_suffix(".json").write_text(json.dumps({"config": config, "history": history}, indent=2))
    print(f"final test errors: {history[-1]['test_errors']} / {len(x_test)}  ->  {args.out}")


if __name__ == "__main__":
    main()
