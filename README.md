# distillation

Experiments on knowledge distillation. Each project lives in its own folder; the uv environment (`pyproject.toml`, `uv.lock`, `.python-version`) at the root is shared.

| Folder | Project |
|---|---|
| [`toy_distillation/`](toy_distillation/README.md) | Reproduction of Ziming Liu, [*A toy model of distillation*](https://kindxiaoming.github.io/blog/2026/distillation/) (2026) |

```bash
uv sync
uv run toy_distillation/run_experiments.py all
```
