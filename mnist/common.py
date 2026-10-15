"""Shared pieces for the MNIST experiments of Hinton, Vinyals & Dean,
"Distilling the Knowledge in a Neural Network" (2015), Section 3:
https://arxiv.org/abs/1503.02531

  * the whole of MNIST is kept in memory as float tensors in [0, 1]
  * jitter: every training image is shifted by up to 2 pixels in each direction
  * MLP: 784 -> hidden -> hidden -> 10 with ReLU, optional dropout on the input
    and hidden layers, and a max-norm constraint on each hidden unit's incoming weights
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

DATA_ROOT = Path(__file__).resolve().parent / "data"


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
    def __init__(self, hidden: int, p_in: float = 0.0, p_hidden: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)
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
