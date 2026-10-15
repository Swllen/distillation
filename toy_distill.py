"""Core of the toy model of distillation.

Reproduces the setup of Ziming Liu, "A toy model of distillation" (2026):
https://kindxiaoming.github.io/blog/2026/distillation/

Teacher and student are both linear maps R^d -> R^V (d=10, V=1000).
The teacher is random, frozen, and its logits are divided by a temperature T.

  * regular : sample a one-hot label y ~ p_teacher(.|x), train with CE(y, student)
  * distill : train with soft CE(p_teacher, student) = KL(p_teacher || p_student) + H(p_teacher)

Both training losses are reported on the same scale, so the teacher entropy
E_x[H(p_teacher)] is the lowest achievable loss (red dashed line in the blog).
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
    method: str = "distill"  # "regular" | "distill"
    optimizer: str = "adam"  # "adam" | "sgd"
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


def make_optimizer(cfg: Config, params) -> torch.optim.Optimizer:
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr)
    raise ValueError(f"unknown optimizer {cfg.optimizer}")


def soft_ce(student_logits: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
    """CE(p_teacher, p_student) = KL + H(p_teacher), averaged over the batch."""
    return -(teacher_probs * F.log_softmax(student_logits, dim=-1)).sum(-1).mean()


@torch.no_grad()
def teacher_stats(teacher: Teacher, x: torch.Tensor) -> dict:
    """Entropy (best achievable loss) and E_x ||p||_2 (predicted speedup = 1/||p||_2)."""
    p = F.softmax(teacher(x), dim=-1)
    entropy = -(p * torch.log(p.clamp_min(1e-30))).sum(-1).mean().item()
    p_norm = p.norm(dim=-1).mean().item()
    return {"entropy": entropy, "p_norm": p_norm, "speedup_pred": 1.0 / p_norm}


def train(cfg: Config) -> dict:
    """Train one student. Returns per-step training loss and periodic clean eval loss.

    train_loss: the loss actually optimised (one-hot CE for regular, soft CE for distill).
    eval_loss : soft CE against the teacher on a fixed held-out set, comparable across methods.
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
    stats = teacher_stats(teacher, x_eval)

    train_loss, eval_steps, eval_loss = [], [], []

    def evaluate(step: int):
        with torch.no_grad():
            eval_steps.append(step)
            eval_loss.append(soft_ce(student(x_eval), p_eval).item())

    evaluate(0)
    for step in range(1, cfg.steps + 1):
        x = torch.randn(cfg.batch_size, cfg.d_in, generator=gx, device=dev)
        with torch.no_grad():
            p = F.softmax(teacher(x), dim=-1)
        logits = student(x)
        if cfg.method == "regular":
            y = torch.multinomial(p, 1, generator=gy).squeeze(-1)
            loss = F.cross_entropy(logits, y)
        elif cfg.method == "distill":
            loss = soft_ce(logits, p)
        else:
            raise ValueError(f"unknown method {cfg.method}")

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
        **stats,
    }
