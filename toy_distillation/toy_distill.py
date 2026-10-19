"""Core of the toy model of distillation.

Reproduces the setup of Ziming Liu, "A toy model of distillation" (2026):
https://kindxiaoming.github.io/blog/2026/distillation/

Teacher and student are both linear maps R^d -> R^V (d=10, V=1000).
The teacher is random, frozen, and its logits are divided by a temperature T.

  * regular : sample a one-hot label y ~ p_teacher(.|x), train with CE(y, student)
  * distill : train with soft CE(p_teacher, student) = KL(p_teacher || p_student) + H(p_teacher)

Both training losses are reported on the same scale, so the teacher entropy
E_x[H(p_teacher)] is the lowest achievable loss (red dashed line in the blog).

Variants of the soft target term (same names as mnist/train_student.py), with q = student, p = teacher:

  * swapped        : -sum q log p = KL(q || p) + H(q)   minimum at q = one-hot(argmax p)
  * reverse_kl     : KL(q || p) = swapped - H(q)        minimum at q = p
  * teacher_argmax : CE(one-hot(argmax p), q)           one-hot training on the teacher's labels
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    d_in: int = 10
    vocab: int = 1000
    temperature: float = 1.0
    method: str = "distill"  # one of METHODS
    optimizer: str = "adam"  # a key of OPTIMIZERS
    lr: float = 1e-2
    batch_size: int = 128
    steps: int = 1000
    teacher_seed: int = 0
    student_seed: int = 1
    data_seed: int = 2
    eval_size: int = 8192
    eval_every: int = 10
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


class Teacher(nn.Module):
    """Random linear teacher (PyTorch default nn.Linear init), logits scaled by 1/T."""

    def __init__(self, d_in: int, vocab: int, temperature: float, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.linear = nn.Linear(d_in, vocab)
        self.temperature = temperature
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) / self.temperature


def make_student(cfg: Config) -> nn.Module:
    torch.manual_seed(cfg.student_seed)
    return nn.Linear(cfg.d_in, cfg.vocab).to(cfg.device)


class SignSGD(torch.optim.Optimizer):
    """theta -= lr * sign(g): the extreme of per-coordinate normalisation."""

    def __init__(self, params, lr: float):
        super().__init__(params, dict(lr=lr))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.add_(p.grad.sign(), alpha=-group["lr"])


# Linear in g (no speedup expected): sgd, momentum.
# Normalised by a running gradient scale (speedup expected): rmsprop, adagrad, adam, adamax, signsgd.
OPTIMIZERS = {
    "sgd": lambda params, lr: torch.optim.SGD(params, lr=lr),
    "momentum": lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.9),
    "rmsprop": lambda params, lr: torch.optim.RMSprop(params, lr=lr),
    "adagrad": lambda params, lr: torch.optim.Adagrad(params, lr=lr),
    "adam": lambda params, lr: torch.optim.Adam(params, lr=lr),
    "adamax": lambda params, lr: torch.optim.Adamax(params, lr=lr),
    "signsgd": lambda params, lr: SignSGD(params, lr=lr),
}


def make_optimizer(cfg: Config, params) -> torch.optim.Optimizer:
    if cfg.optimizer not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {cfg.optimizer}; choose from {sorted(OPTIMIZERS)}")
    return OPTIMIZERS[cfg.optimizer](params, cfg.lr)


def soft_ce(student_logits: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
    """CE(p_teacher, p_student) = KL + H(p_teacher), averaged over the batch."""
    return -(teacher_probs * F.log_softmax(student_logits, dim=-1)).sum(-1).mean()


METHODS = ("regular", "distill", "swapped", "reverse_kl", "teacher_argmax")


def method_loss(method: str, logits: torch.Tensor, teacher_logits: torch.Tensor,
                gy: torch.Generator) -> torch.Tensor:
    """Training loss of `method` for one batch; teacher_logits are already divided by T."""
    if method == "regular":
        p = F.softmax(teacher_logits, dim=-1)
        y = torch.multinomial(p, 1, generator=gy).squeeze(-1)
        return F.cross_entropy(logits, y)
    if method == "distill":
        return soft_ce(logits, F.softmax(teacher_logits, dim=-1))
    if method == "teacher_argmax":
        return F.cross_entropy(logits, teacher_logits.argmax(dim=-1))
    log_q = F.log_softmax(logits, dim=-1)
    log_p = F.log_softmax(teacher_logits, dim=-1)
    if method == "swapped":
        return -(log_q.exp() * log_p).sum(-1).mean()
    if method == "reverse_kl":
        return (log_q.exp() * (log_q - log_p)).sum(-1).mean()
    raise ValueError(f"unknown method {method}; choose from {METHODS}")


@torch.no_grad()
def teacher_stats(teacher: Teacher, x: torch.Tensor) -> dict:
    """Entropy (best achievable loss) and E_x ||p||_2 (predicted speedup = 1/||p||_2)."""
    p = F.softmax(teacher(x), dim=-1)
    entropy = -(p * torch.log(p.clamp_min(1e-30))).sum(-1).mean().item()
    p_norm = p.norm(dim=-1).mean().item()
    return {"entropy": entropy, "p_norm": p_norm, "speedup_pred": 1.0 / p_norm}


def train(cfg: Config) -> dict:
    """Train one student. Returns per-step training loss and periodic clean eval metrics.

    train_loss  : the loss actually optimised (differs between methods).
    eval_loss   : soft CE(p, q) against the teacher on a fixed held-out set, comparable across methods.
    eval_acc    : fraction of eval points where argmax q == argmax p.
    eval_entropy: mean student entropy H(q); collapses towards 0 if q becomes one-hot.
    """
    dev = cfg.device
    teacher = Teacher(cfg.d_in, cfg.vocab, cfg.temperature, cfg.teacher_seed).to(dev)
    student = make_student(cfg)
    opt = make_optimizer(cfg, student.parameters())

    # Separate generators so that x batches are identical across methods.
    gx = torch.Generator(device=dev).manual_seed(cfg.data_seed)
    gy = torch.Generator(device=dev).manual_seed(cfg.data_seed + 10_000)
    geval = torch.Generator(device=dev).manual_seed(12345)
    x_eval = torch.randn(cfg.eval_size, cfg.d_in, generator=geval, device=dev)
    with torch.no_grad():
        p_eval = F.softmax(teacher(x_eval), dim=-1)
        top_eval = p_eval.argmax(dim=-1)
    stats = teacher_stats(teacher, x_eval)

    train_loss, eval_steps, eval_loss, eval_acc, eval_entropy = [], [], [], [], []

    def evaluate(step: int):
        with torch.no_grad():
            logits = student(x_eval)
            log_q = F.log_softmax(logits, dim=-1)
            eval_steps.append(step)
            eval_loss.append(-(p_eval * log_q).sum(-1).mean().item())
            eval_acc.append((logits.argmax(dim=-1) == top_eval).float().mean().item())
            eval_entropy.append(-(log_q.exp() * log_q).sum(-1).mean().item())

    evaluate(0)
    for step in range(1, cfg.steps + 1):
        x = torch.randn(cfg.batch_size, cfg.d_in, generator=gx, device=dev)
        with torch.no_grad():
            teacher_logits = teacher(x)
        loss = method_loss(cfg.method, student(x), teacher_logits, gy)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        train_loss.append(loss.item())
        if step % cfg.eval_every == 0 or step == cfg.steps:
            evaluate(step)

    return {
        "config": cfg.__dict__.copy(),
        "train_loss": train_loss,
        "eval_steps": eval_steps,
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "eval_entropy": eval_entropy,
        **stats,
    }
