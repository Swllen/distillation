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
import wandb
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


@torch.no_grad()
def evaluate(model: MLP, x: torch.Tensor, y: torch.Tensor, batch_size: int = 2000) -> dict[str, float]:
    """Errors, mean cross-entropy and the fraction of dead hidden units (never active on x)."""
    model.eval()
    errors, loss = 0, 0.0
    alive1 = torch.zeros(model.fc1.out_features, dtype=torch.bool, device=x.device)
    alive2 = torch.zeros(model.fc2.out_features, dtype=torch.bool, device=x.device)
    for i in range(0, len(x), batch_size):
        h1 = F.relu(model.fc1(x[i : i + batch_size].flatten(start_dim=1)))
        h2 = F.relu(model.fc2(h1))
        logits = model.fc3(h2)
        yb = y[i : i + batch_size]
        errors += int((logits.argmax(dim=1) != yb).sum())
        loss += float(F.cross_entropy(logits, yb, reduction="sum"))
        alive1 |= (h1 > 0).any(dim=0)
        alive2 |= (h2 > 0).any(dim=0)
    return {
        "errors": errors,
        "loss": loss / len(x),
        "dead_units_fc1": 1 - alive1.float().mean().item(),
        "dead_units_fc2": 1 - alive2.float().mean().item(),
    }


@torch.no_grad()
def weight_stats(model: MLP, max_norm: float) -> dict[str, float]:
    """Largest incoming-weight norm of a hidden unit per layer, and the fraction of units at the max-norm cap."""
    out = {}
    for name in ("fc1", "fc2"):
        norms = getattr(model, name).weight.norm(dim=1)
        out[f"{name}_max_row_norm"] = norms.max().item()
        if max_norm > 0:
            out[f"{name}_frac_at_max_norm"] = (norms >= max_norm * (1 - 1e-4)).float().mean().item()
    return out


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
    p.add_argument("--log-every", type=int, help=f"steps between W&B training logs (default {OPTIONAL['log_every']})")
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, help="log to Weights & Biases (default on)")
    p.add_argument("--wandb-project", help=f"default {OPTIONAL['wandb_project']}")
    p.add_argument("--wandb-entity", help="W&B user or team; default your own account")
    p.add_argument("--name", help="W&B run name; default the config file name")
    p.add_argument("--optimizer", choices=("momentum", "adam"), help="momentum: the update of [5] (default); adam: torch.optim.Adam")
    p.add_argument("--adam-beta1", type=float)
    p.add_argument("--adam-beta2", type=float)


# Settings with a default, so config files written before they existed keep working.
OPTIONAL = {
    "log_every": 100,
    "wandb": True,
    "wandb_project": "distillation-mnist",
    "wandb_entity": "",
    "name": "",
    "optimizer": "momentum",
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
}


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
    for key, value in OPTIONAL.items():
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


def start_run(args: argparse.Namespace, job: str, paper_test_errors: int | None = None):
    """Start a W&B run (a no-op run with --no-wandb). Every metric is plotted against epoch."""
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.name or args.config.stem,
        group=args.config.stem,
        job_type=job,
        config=config_dict(args),
        mode=None if args.wandb else "disabled",  # None: online, or whatever WANDB_MODE says
    )
    run.define_metric("epoch")
    run.define_metric("*", step_metric="epoch")
    if paper_test_errors is not None:
        run.summary["paper_test_errors"] = paper_test_errors
    return run


# loss_fn(logits, images, labels) -> loss terms; the "loss" entry is minimized, all are logged.
LossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]


def train(model: MLP, args: argparse.Namespace, data: tuple[torch.Tensor, ...], loss_fn: LossFn, run) -> list[dict]:
    """Train and return the evaluation history.

    Every step: jitter the minibatch, compute the loss terms, take one optimizer step,
    then project each hidden unit's incoming weights onto the max-norm ball. The
    learning rate is eps_t = lr * lr_decay^t (t = epoch) for both optimizers.
      optimizer = "momentum":  the update of [5],
          dw_t = p_t dw_{t-1} - (1 - p_t) eps_t <grad>,  w_t = w_{t-1} + dw_t,
          with momentum p_t ramped from momentum_initial to momentum_final
      optimizer = "adam":      torch.optim.Adam with betas (adam_beta1, adam_beta2);
          the momentum_* settings are ignored

    Logged to W&B: the loss terms and gradient norm averaged over every log_every steps,
    the learning rate (and momentum), and every eval_every epochs the train and test errors
    and loss, the fraction of dead hidden units, and the hidden-unit weight norms.
    """
    x_train, y_train, x_test, y_test = data
    n = len(x_train)
    steps_per_epoch = n // args.batch_size
    params = list(model.parameters())
    adam = args.optimizer == "adam"
    if adam:
        opt = torch.optim.Adam(params, lr=args.lr, betas=(args.adam_beta1, args.adam_beta2))
    else:
        deltas = [torch.zeros_like(q) for q in params]

    print(f"device={args.device}  params={sum(q.numel() for q in params):,}  steps={args.epochs * steps_per_epoch:,}")
    run.summary["params"] = sum(q.numel() for q in params)
    history = []
    start = time.time()
    sums: dict[str, torch.Tensor] = {}
    window = 0
    for epoch in range(args.epochs):
        eps, mom = lr_at(args, epoch), momentum_at(args, epoch)
        if adam:
            for group in opt.param_groups:
                group["lr"] = eps
        model.train()
        perm = torch.randperm(n, device=args.device)
        epoch_loss = torch.zeros((), device=args.device)
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch_size : (i + 1) * args.batch_size]
            x = jitter(x_train[idx], args.jitter)
            terms = loss_fn(model(x), x, y_train[idx])
            grads = torch.autograd.grad(terms["loss"], params)
            with torch.no_grad():
                terms["grad_norm"] = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
                if adam:
                    for q, g in zip(params, grads):
                        q.grad = g
                    opt.step()
                else:
                    torch._foreach_mul_(deltas, mom)
                    torch._foreach_add_(deltas, grads, alpha=-(1 - mom) * eps)
                    torch._foreach_add_(params, deltas)
            if args.max_norm > 0:
                model.apply_max_norm(args.max_norm)

            for k, v in terms.items():
                sums[k] = sums.get(k, 0) + v.detach()
            window += 1
            epoch_loss += terms["loss"].detach()
            step = epoch * steps_per_epoch + i + 1
            if step % args.log_every == 0:
                log = {f"train/{k}": (v / window).item() for k, v in sums.items()}
                log.update({"epoch": step / steps_per_epoch, "optim/lr": eps})
                if not adam:
                    log.update({"optim/effective_lr": (1 - mom) * eps, "optim/momentum": mom})
                run.log(log, step=step)
                sums.clear()
                window = 0

        done = epoch + 1
        if done % args.eval_every == 0 or done == args.epochs:
            test = evaluate(model, x_test, y_test)
            train_eval = evaluate(model, x_train, y_train)
            mean_loss = epoch_loss.item() / steps_per_epoch
            history.append({"epoch": done, "train_loss": mean_loss, "test_errors": test["errors"], "train_errors": train_eval["errors"]})
            log = {"epoch": done, "train/epoch_loss": mean_loss, "time/elapsed_s": time.time() - start}
            log.update({f"test/{k}": v for k, v in test.items()})
            log.update({"train/errors": train_eval["errors"], "train/eval_loss": train_eval["loss"]})
            log.update({f"weights/{k}": v for k, v in weight_stats(model, args.max_norm).items()})
            run.log(log, step=done * steps_per_epoch)
            optim_info = f"lr {eps:.2e}" if adam else f"eps {eps:.4f}  momentum {mom:.3f}"
            print(
                f"epoch {done:4d}  loss {mean_loss:.4f}  test errors {test['errors']:4d}  {optim_info}  {time.time() - start:.0f}s",
                flush=True,
            )
    return history


def save_run(model: MLP, args: argparse.Namespace, history: list[dict], run) -> None:
    config = config_dict(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config, "history": history}, args.out)
    args.out.with_suffix(".json").write_text(json.dumps({"config": config, "history": history}, indent=2))
    run.summary["final_test_errors"] = history[-1]["test_errors"]
    run.summary["final_train_errors"] = history[-1]["train_errors"]
    run.summary["checkpoint"] = str(args.out)
    run.finish()
    print(f"final test errors: {history[-1]['test_errors']} / 10000  ->  {args.out}")


def load_model(path: Path, device: str | torch.device) -> MLP:
    """Load a checkpoint written by save_run, in eval mode."""
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = MLP(cfg["hidden"], bias=cfg.get("bias", True)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()
