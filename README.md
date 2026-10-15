# distillation

Experiments on knowledge distillation. Each project lives in its own folder; the uv environment (`pyproject.toml`, `uv.lock`, `.python-version`) at the root is shared.

| Folder | Project |
|---|---|
| [`toy_distillation/`](toy_distillation/README.md) | Reproduction of Ziming Liu, [*A toy model of distillation*](https://kindxiaoming.github.io/blog/2026/distillation/) (2026) |
| [`mnist/`](mnist/README.md) | MNIST experiment of Hinton, Vinyals & Dean, [*Distilling the Knowledge in a Neural Network*](https://arxiv.org/abs/1503.02531) (2015) |

```bash
uv sync
uv run toy_distillation/run_experiments.py all
```
