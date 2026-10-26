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
import math
import os
import tempfile
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


def load_data(args) -> tuple[torch.Tensor, ...]:
    """load_mnist, with the digits in args.exclude_digits removed from the training set
    (for distillation, the transfer set). The test set stays complete."""
    x_train, y_train, x_test, y_test = load_mnist(args.device)
    if args.exclude_digits:
        keep = ~torch.isin(y_train, torch.tensor(args.exclude_digits, device=y_train.device))
        x_train, y_train = x_train[keep], y_train[keep]
        print(f"training set without digits {args.exclude_digits}: {len(x_train)} images")
    return x_train, y_train, x_test, y_test


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
def evaluate(model: MLP, x: torch.Tensor, y: torch.Tensor, loss_fn, batch_size: int = 2000) -> dict:
    """Errors (in total and per true class), the training loss (loss_fn's "loss", in eval
    mode and without jitter) and the fraction of dead hidden units (never active on x)."""
    model.eval()
    loss = 0.0
    errors_by_class = torch.zeros(10, dtype=torch.long, device=x.device)
    alive1 = torch.zeros(model.fc1.out_features, dtype=torch.bool, device=x.device)
    alive2 = torch.zeros(model.fc2.out_features, dtype=torch.bool, device=x.device)
    for i in range(0, len(x), batch_size):
        h1 = F.relu(model.fc1(x[i : i + batch_size].flatten(start_dim=1)))
        h2 = F.relu(model.fc2(h1))
        logits = model.fc3(h2)
        xb, yb = x[i : i + batch_size], y[i : i + batch_size]
        errors_by_class += torch.bincount(yb[logits.argmax(dim=1) != yb], minlength=10)
        loss += float(loss_fn(logits, xb, yb)["loss"]) * len(xb)
        alive1 |= (h1 > 0).any(dim=0)
        alive2 |= (h2 > 0).any(dim=0)
    return {
        "errors": int(errors_by_class.sum()),
        "errors_by_class": errors_by_class.tolist(),
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
    p.add_argument("--lr-schedule", choices=("exponential", "cosine"), help="exponential: lr * lr_decay^epoch (default); cosine: anneal lr -> lr_min")
    p.add_argument("--lr-min", type=float, help="final learning rate of the cosine schedule (default 0)")
    p.add_argument("--exclude-digits", type=int, nargs="*", help="digits removed from the training set, e.g. 3 (default none)")


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
    "lr_schedule": "exponential",
    "lr_min": 0.0,
    "exclude_digits": [],
    "soft_loss": "forward",  # train_student.py only
    "opal": False,  # train_student.py only, as are the opal_* settings below
    "opal_backend": "claude",
    "opal_max_cost_usd": 0.0,  # 0: no cap
    "opal_decisions": 0,  # 0: OPAL's default
    "opal_shell": True,  # False: the agent gets file tools only, no shell
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
        if key in vars(args) and getattr(args, key) is None:
            setattr(args, key, value)
    missing = [k for k, v in vars(args).items() if v is None and k not in ("out", "device")]
    if missing:
        p.error(f"missing setting(s), add them to {args.config}: {', '.join(missing)}")
    if args.out is None:
        args.out = HERE / "checkpoints" / f"{args.config.stem}.pt"
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def args_from_dict(
    p: argparse.ArgumentParser, settings: dict, name: str, out: Path | None = None, device: str | None = None
) -> argparse.Namespace:
    """Like parse_args, but the settings come from a dict instead of a config file; `name`
    stands in for the config file name (W&B run name and group, default checkpoint name)."""
    args = p.parse_args([])
    unknown = sorted(set(settings) - set(vars(args)))
    if unknown:
        raise ValueError(f"unknown setting(s): {', '.join(unknown)}")
    for key, value in {**{k: v for k, v in OPTIONAL.items() if k in vars(args)}, **settings}.items():
        setattr(args, key, value)
    missing = [k for k, v in vars(args).items() if v is None and k not in ("out", "device")]
    if missing:
        raise ValueError(f"missing setting(s): {', '.join(missing)}")
    args.config = Path(f"{name}.toml")
    args.out = Path(out) if out is not None else HERE / "checkpoints" / f"{name}.pt"
    args.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
    """Learning rate for an epoch (counted from 0); constant within the epoch."""
    if args.lr_schedule == "cosine":
        return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * epoch / args.epochs))
    return args.lr * args.lr_decay**epoch


def momentum_at(args: argparse.Namespace, epoch: int) -> float:
    if epoch >= args.momentum_epochs:
        return args.momentum_final
    return args.momentum_initial + (epoch / args.momentum_epochs) * (args.momentum_final - args.momentum_initial)


class HintonMomentum(torch.optim.Optimizer):
    """The momentum update of [5] as a torch optimizer (so that OPAL can wrap it):
    dw_t = p dw_{t-1} - (1 - p) eps <grad>,  w_t = w_{t-1} + dw_t,
    with eps = group["lr"] and p = group["momentum"], both set every epoch by EpochSchedule."""

    def __init__(self, params, lr: float, momentum: float):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            params = [q for q in group["params"] if q.grad is not None]
            deltas = []
            for q in params:
                if "delta" not in self.state[q]:
                    self.state[q]["delta"] = torch.zeros_like(q)
                deltas.append(self.state[q]["delta"])
            mom, eps = group["momentum"], group["lr"]
            torch._foreach_mul_(deltas, mom)
            torch._foreach_add_(deltas, [q.grad for q in params], alpha=-(1 - mom) * eps)
            torch._foreach_add_(params, deltas)


class EpochSchedule:
    """Sets every group's lr to lr_at and, for HintonMomentum, its momentum to momentum_at.
    Call step() at the end of each epoch."""

    def __init__(self, optimizer: torch.optim.Optimizer, args: argparse.Namespace):
        self.optimizer, self.args, self.epoch = optimizer, args, 0
        self._apply()

    def step(self) -> None:
        self.epoch += 1
        self._apply()

    def _apply(self) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr_at(self.args, self.epoch)
            if "momentum" in group:
                group["momentum"] = momentum_at(self.args, self.epoch)

    def state_dict(self) -> dict:
        return {"epoch": self.epoch}

    def load_state_dict(self, state: dict) -> None:
        self.epoch = state["epoch"]


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


def train(
    model: MLP, args: argparse.Namespace, data: tuple[torch.Tensor, ...], loss_fn: LossFn, run, opal_goal: str = ""
) -> list[dict]:
    """Train and return the evaluation history. With opal_goal (train_student.py --opal) the
    optimizer is wrapped by OPAL; see opal_wrap.

    Every step: jitter the minibatch, compute the loss terms, take one optimizer step,
    then project each hidden unit's incoming weights onto the max-norm ball. The
    learning rate eps_t comes from lr_at: lr * lr_decay^t, or with lr_schedule = "cosine"
    lr_min + (lr - lr_min) (1 + cos(pi t / epochs)) / 2  (t = epoch), for both optimizers.
      optimizer = "momentum":  the update of [5],
          dw_t = p_t dw_{t-1} - (1 - p_t) eps_t <grad>,  w_t = w_{t-1} + dw_t,
          with momentum p_t ramped from momentum_initial to momentum_final
      optimizer = "adam":      torch.optim.Adam with betas (adam_beta1, adam_beta2);
          the momentum_* settings are ignored

    Logged to W&B: the loss terms and gradient norm averaged over every log_every steps,
    the learning rate (and momentum), and every eval_every epochs the train and test errors,
    the training loss on both sets (train/eval_loss, test/loss; eval mode, no jitter), the
    fraction of dead hidden units, and the
    hidden-unit weight norms.
    """
    x_train, y_train, x_test, y_test = data
    n = len(x_train)
    steps_per_epoch = n // args.batch_size
    param_names, params = zip(*model.named_parameters())
    params = list(params)
    adam = args.optimizer == "adam"
    if adam:
        opt = torch.optim.Adam(params, lr=args.lr, betas=(args.adam_beta1, args.adam_beta2))
    else:
        opt = HintonMomentum(params, lr=args.lr, momentum=args.momentum_initial)
    schedule = EpochSchedule(opt, args)
    if opal_goal:
        opt = opal_wrap(opt, schedule, model, args, n, steps_per_epoch, opal_goal)

    print(f"device={args.device}  params={sum(q.numel() for q in params):,}  steps={args.epochs * steps_per_epoch:,}")
    run.summary["params"] = sum(q.numel() for q in params)
    history = []
    start = time.time()
    sums: dict[str, torch.Tensor] = {}
    window = 0
    for epoch in range(args.epochs):
        eps, mom = lr_at(args, epoch), momentum_at(args, epoch)  # the schedule's values; OPAL's agent may change them
        model.train()
        perm = torch.randperm(n, device=args.device)
        epoch_sums: dict[str, torch.Tensor] = {}
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch_size : (i + 1) * args.batch_size]
            x = jitter(x_train[idx], args.jitter)
            logits = model(x)
            terms = loss_fn(logits, x, y_train[idx])
            *grads, logit_grad = torch.autograd.grad(terms["loss"], params + [logits])
            with torch.no_grad():
                layer_norms = torch._foreach_norm(grads)
                terms["grad_norm"] = torch.linalg.vector_norm(torch.stack(layer_norms))
                for name, g in zip(param_names, layer_norms):
                    terms[f"grad_{name}"] = g
                # Per-image gradient on the output logits (the loss is a batch mean, hence * batch size).
                terms["logit_grad_norm"] = logit_grad.norm(dim=1).mean() * len(x)
            for q, g in zip(params, grads):
                q.grad = g
            if opal_goal:
                opt.step(loss=terms["loss"])  # OPAL shows the agent the training loss
            else:
                opt.step()
            if args.max_norm > 0:
                model.apply_max_norm(args.max_norm)

            for k, v in terms.items():
                sums[k] = sums.get(k, 0) + v.detach()
                epoch_sums[k] = epoch_sums.get(k, 0) + v.detach()
            window += 1
            step = epoch * steps_per_epoch + i + 1
            if step % args.log_every == 0:
                log = {f"train/{k}": (v / window).item() for k, v in sums.items()}
                log.update({"epoch": step / steps_per_epoch, "optim/lr": eps})
                if not adam:
                    log.update({"optim/effective_lr": (1 - mom) * eps, "optim/momentum": mom})
                run.log(log, step=step)
                sums.clear()
                window = 0

        schedule.step()
        done = epoch + 1
        if done % args.eval_every == 0 or done == args.epochs:
            test = evaluate(model, x_test, y_test, loss_fn)
            test_by_class = test.pop("errors_by_class")
            train_eval = evaluate(model, x_train, y_train, loss_fn)
            epoch_means = {k: v.item() / steps_per_epoch for k, v in epoch_sums.items()}
            mean_loss = epoch_means["loss"]
            # Epoch means of every loss term (train_loss, train_soft_kl, ...) plus the eval results.
            entry = {"epoch": done, **{f"train_{k}": v for k, v in epoch_means.items()}}
            entry.update({"train_eval_loss": train_eval["loss"], "test_loss": test["loss"]})
            entry.update({"test_errors": test["errors"], "train_errors": train_eval["errors"], "lr": eps})
            if not adam:
                entry["momentum"] = mom
            excluded_info = ""
            if args.exclude_digits:
                # Test errors on the digits the model never saw in training.
                entry["test_errors_excluded"] = sum(test_by_class[d] for d in args.exclude_digits)
                excluded_info = f" ({entry['test_errors_excluded']} on {args.exclude_digits})"
            history.append(entry)
            if opal_goal:  # recorded in OPAL's run folder and report; the agent does not see it
                opt.log(test_errors=test["errors"], test_loss=test["loss"], train_errors=train_eval["errors"])
            log = {"epoch": done, "train/epoch_loss": mean_loss, "time/elapsed_s": time.time() - start}
            log.update({f"test/{k}": v for k, v in test.items()})
            log.update({f"test/errors_on_{d}": test_by_class[d] for d in range(10)})
            log.update({"train/errors": train_eval["errors"], "train/eval_loss": train_eval["loss"]})
            log.update({f"weights/{k}": v for k, v in weight_stats(model, args.max_norm).items()})
            run.log(log, step=done * steps_per_epoch)
            optim_info = f"lr {eps:.2e}" if adam else f"eps {eps:.4f}  momentum {mom:.3f}"
            grad_info = f"grad {epoch_means['grad_norm']:.2e}  logit grad {epoch_means['logit_grad_norm']:.2e}"
            print(
                f"epoch {done:4d}  loss {mean_loss:.4f}  test loss {test['loss']:.4f}  "
                f"test errors {test['errors']:4d}{excluded_info}  "
                f"{grad_info}  {optim_info}  {time.time() - start:.0f}s",
                flush=True,
            )
    if opal_goal:
        opt.close()
    return history


# Linux limits a Unix socket path to 107 characters; the sandbox of the agent's shell makes sockets
# such as srt-mux-<pid>-<n>.sock (up to about 30 characters) in the run folder's agent-tmp/.
SOCKET_PATH_MAX = 107
SOCKET_NAME_MAX = 30


def short_agent_tmp(run_dir: Path) -> None:
    """Make <run_dir>/agent-tmp a link to a new private folder under /tmp/claude-<uid> (Claude Code's own
    temp folder), so the sandbox's sockets get a short path; the records stay in run_dir. OPAL 0.3.8
    resolves the link (backends/claude.py). Linux only: elsewhere the agent's shell has no sandbox."""
    if not hasattr(os, "getuid"):
        return
    base = Path(tempfile.gettempdir()) / f"claude-{os.getuid()}"
    base.mkdir(mode=0o700, exist_ok=True)
    target = tempfile.mkdtemp(prefix="opal-", dir=base)  # mode 0700, unique name
    if len(target) + 1 + SOCKET_NAME_MAX > SOCKET_PATH_MAX:
        raise RuntimeError(f"short_agent_tmp: {target} is too long for the sandbox sockets")
    (run_dir / "agent-tmp").symlink_to(target)


def opal_wrap(opt, schedule, model: MLP, args: argparse.Namespace, n: int, steps_per_epoch: int, goal: str):
    """Wrap the optimizer with OPAL (https://github.com/xingjian-bai/opal): a coding agent rewrites the
    optimizer program while the model trains. The starting program calls opt.step() and schedule.step()
    unchanged, so with backend "keep" (no agent) the run matches the plain one bit for bit.
    Records go to opal_runs/<checkpoint name>/; train_student.py's --opal-* flags set the options."""
    import opal

    run_dir = HERE / "opal_runs" / args.out.stem
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists; pass another --out (or --name and --out) for a new OPAL run")
    run_dir.mkdir(parents=True)
    options = dict(epochs=args.epochs, train_size=n, batch_size=args.batch_size, same_order_each_epoch=False)
    if args.opal_backend == "claude":
        short_agent_tmp(run_dir)
        if not args.opal_shell:
            # On a Linux host whose AppArmor denies nested user namespaces inside bubblewrap (Ubuntu's
            # unpriv_bwrap profile), the sandbox starts but every shell command fails.
            options["backend_options"] = dict(sandbox=False)
    if args.opal_max_cost_usd > 0:
        options["max_cost_usd"] = args.opal_max_cost_usd
    if args.opal_decisions > 0:
        options["decisions"] = args.opal_decisions
    return opal.wrap(
        opt,
        model,
        scheduler=schedule,
        total_steps=args.epochs * steps_per_epoch,
        goal=goal,
        backend=args.opal_backend,
        run_dir=run_dir,
        **options,
    )


def save_run(model: MLP, args: argparse.Namespace, history: list[dict], run) -> None:
    config = config_dict(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config, "history": history}, args.out)
    args.out.with_suffix(".json").write_text(json.dumps({"config": config, "history": history}, indent=2))
    run.summary["final_test_errors"] = history[-1]["test_errors"]
    run.summary["final_train_errors"] = history[-1]["train_errors"]
    if "test_errors_excluded" in history[-1]:
        run.summary["final_test_errors_excluded"] = history[-1]["test_errors_excluded"]
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
