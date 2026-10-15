"""Shared pieces for the MNIST experiments of Hinton, Vinyals & Dean,
"Distilling the Knowledge in a Neural Network" (2015), Section 3:
https://arxiv.org/abs/1503.02531

  * the whole of MNIST is kept in memory as float tensors in [0, 1]
  * jitter: every training image is shifted by up to 2 pixels in each direction
  * MLP: 784 -> hidden -> hidden -> 10 with ReLU, optional dropout on the input
    and hidden layers, and a max-norm constraint on each hidden unit's incoming weights
  * settings come from TOML files in configs/, overridable by command-line flags
  * training uses the momentum update and schedules of the paper's reference [5],
    Hinton et al. (2012), arXiv:1207.0580, Appendix A
"""

from __future__ import annotations

import argparse
import json
import time
import tomllib
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "data"


def load_mnist(device: str | torch.device) -> tuple[torch.Tensor, ...]:
    """Return (x_train, y_train, x_test, y_test); images are (N, 28, 28) floats in [0, 1]."""
    out = []
    for train in (True, False):
        ds = datasets.MNIST(root=DATA_ROOT, train=train, download=True)
        out += [(ds.data.float() / 255.0).to(device), ds.targets.to(device)]
    return tuple(out)


def jitter(x: torch.Tensor, max_shift: int = 2) -> torch.Tensor:
    """Shift each image in a (B, 28, 28) batch by an independent random (dy, dx) in
    [-max_shift, max_shift], filling the uncovered border with zeros."""
    if max_shift == 0:
        return x
    b, h, w = x.shape
    padded = F.pad(x, (max_shift,) * 4)
    dy = torch.randint(0, 2 * max_shift + 1, (b, 1), device=x.device)
    dx = torch.randint(0, 2 * max_shift + 1, (b, 1), device=x.device)
    rows = (torch.arange(h, device=x.device) + dy)[:, :, None]
    cols = (torch.arange(w, device=x.device) + dx)[:, None, :]
    return padded[torch.arange(b, device=x.device)[:, None, None], rows, cols]


class MLP(nn.Module):
    def __init__(self, hidden: int, p_in: float = 0.0, p_hidden: float = 0.0, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden, bias=bias)
        self.fc2 = nn.Linear(hidden, hidden, bias=bias)
        self.fc3 = nn.Linear(hidden, 10, bias=bias)
        self.drop_in = nn.Dropout(p_in)
        self.drop_hidden = nn.Dropout(p_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop_in(x.flatten(start_dim=1))
        x = self.drop_hidden(F.relu(self.fc1(x)))
        x = self.drop_hidden(F.relu(self.fc2(x)))
        return self.fc3(x)

    @torch.no_grad()
    def apply_max_norm(self, c: float) -> None:
        """Rescale each hidden unit's incoming weight vector to have L2 norm at most c."""
        for fc in (self.fc1, self.fc2):
            fc.weight.mul_(fc.weight.norm(dim=1, keepdim=True).clamp(min=c).reciprocal().mul(c))


@torch.no_grad()
def logits_of(model: nn.Module, x: torch.Tensor, batch_size: int = 2000) -> torch.Tensor:
    model.eval()
    return torch.cat([model(x[i : i + batch_size]) for i in range(0, len(x), batch_size)])


def test_errors(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> int:
    """Number of misclassified examples (the paper reports test errors out of 10000)."""
    return int((logits_of(model, x).argmax(dim=1) != y).sum())


# ---- shared training code (config files, optimizer of [5], training loop) ----


def add_training_args(p: argparse.ArgumentParser, default_config: Path) -> None:
    """Flags shared by the teacher and student scripts. Every setting defaults to None
    and is filled in from the TOML config file by parse_args."""
    p.add_argument("--config", type=Path, default=default_config)
    p.add_argument("--out", type=Path, help="checkpoint path; default checkpoints/<config name>.pt")
    p.add_argument("--device", help="default: cuda if available, else cpu")
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


def parse_args(p: argparse.ArgumentParser, argv: list[str] | None = None) -> argparse.Namespace:
    """Parse flags, then fill every setting not given on the command line from --config."""
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


def config_dict(args: argparse.Namespace) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}


def make_model(args: argparse.Namespace) -> MLP:
    """MLP with weights ~ N(0, init_std^2) and zero biases, as in [5]."""
    model = MLP(args.hidden, args.p_in, args.p_hidden, bias=args.bias).to(args.device)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=args.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model


def lr_at(args: argparse.Namespace, epoch: int) -> float:
    return args.lr * args.lr_decay**epoch


def momentum_at(args: argparse.Namespace, epoch: int) -> float:
    if epoch >= args.momentum_epochs:
        return args.momentum_final
    return args.momentum_initial + (epoch / args.momentum_epochs) * (args.momentum_final - args.momentum_initial)


def train(
    model: MLP,
    args: argparse.Namespace,
    data: tuple[torch.Tensor, ...],
    loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> list[dict]:
    """Train with the optimizer of [5] and return the test-error history.

    Every step: jitter the minibatch, compute loss_fn(student_logits, images, labels),
    then  dw_t = p_t dw_{t-1} - (1 - p_t) eps_t <grad>,  w_t = w_{t-1} + dw_t,
    then project each hidden unit's incoming weights onto the max-norm ball.
    """
    x_train, y_train, x_test, y_test = data
    n = len(x_train)
    steps_per_epoch = n // args.batch_size
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
            x = jitter(x_train[idx], args.jitter)
            loss = loss_fn(model(x), x, y_train[idx])
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
    return history


def save_run(model: MLP, args: argparse.Namespace, history: list[dict]) -> None:
    config = config_dict(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config, "history": history}, args.out)
    args.out.with_suffix(".json").write_text(json.dumps({"config": config, "history": history}, indent=2))
    print(f"final test errors: {history[-1]['test_errors']} / 10000  ->  {args.out}")


def load_model(path: Path, device: str | torch.device) -> MLP:
    """Load a checkpoint written by save_run, in eval mode."""
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = MLP(cfg["hidden"], bias=cfg.get("bias", True)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()
