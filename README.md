# Toy model of distillation — reproduction

Reproduces Ziming Liu, [*A toy model of distillation*](https://kindxiaoming.github.io/blog/2026/distillation/) (2026).

## Setup

- Teacher and student are both `nn.Linear(10, 1000)` with PyTorch's default init. The teacher is random, frozen, and its logits are divided by a temperature `T`.
- Inputs: `x ~ N(0, I_10)`, sampled fresh every step.
- **Regular**: sample a one-hot label `y ~ p_teacher(·|x)` and train with cross-entropy.
- **Distillation**: train with soft CE `-Σ p_teacher log p_student` = KL + H(p_teacher).
- Adam, lr `1e-2`, batch size `128` by default. The blog does not state lr or batch size, so these are guesses; change them with `--lr` and `--batch-size`.
- The red dashed line is the teacher entropy `E_x H(p_teacher)`, the lowest loss either method can reach.

Prediction from the blog: with Adam, distillation speeds up the burn-in stage by a factor of `1 / ||p||_2`. With SGD there is no speedup.

## Running

The environment is managed with [uv](https://docs.astral.sh/uv/). Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`, and the Python version (3.11) is set in `.python-version`.

```bash
uv sync                                         # create .venv and install the locked versions
uv run run_experiments.py all                   # every experiment, outputs in results/
uv run run_experiments.py temp speedup          # only some of them
uv run run_experiments.py all --batch-size 64 --lr 3e-3 --out results_bs64
```

The script uses CUDA when it is available and the CPU otherwise. The model is tiny, so the CPU works too.

On Linux, the PyPI `torch` wheel is built against a recent CUDA version. If the server's NVIDIA driver is too old for that build, `torch.cuda.is_available()` returns False and the script falls back to the CPU. To use a specific CUDA build instead, add a PyTorch index to `pyproject.toml` (replace `cu126` with a version your driver supports) and then run `uv lock && uv sync`:

```toml
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu126" }
```

| Experiment | What it does | Output |
|---|---|---|
| `temp` | Adam, T ∈ {1, 0.1, 0.01}, 1000 steps (blog fig. 1) | `results/temp.png` |
| `speedup` | Adam, T = 0.1, 3000 steps, log x-axis, raw and with the distillation x-axis rescaled by 1/‖p‖₂ (blog fig. 2) | `results/speedup.png` |
| `sgd` | SGD, T ∈ {1, 0.1, 0.01}, 3000 steps (blog fig. 3) | `results/sgd_temp.png` |
| `sweep` | Adam over 9 temperatures: measured burn-in speedup vs predicted 1/‖p‖₂ | `results/sweep.png` |
| `batch` | The `sweep` repeated for batch sizes 1, 8, 32, 128, 512 (change them with `--batch-sizes`) | `results/batch_sweep.png` |

`all` runs all five experiments. On an RTX PRO 6000 the first four together take about 30 seconds; `batch` adds about 2–3 minutes, since it runs 90 training runs with an eval at every step.

**SGD learning rate.** The `sgd` experiment uses `--sgd-lr`, which defaults to the same value as `--lr` (1e-2). At 1e-2, SGD descends much more slowly than in the blog's figure: at T = 0.1 over 3000 steps, the loss goes from 6.9 to 6.1 here and from about 6.5 to about 4.3 in the blog. The blog probably used a larger SGD lr; try `uv run run_experiments.py sgd --sgd-lr 0.1 --out results_sgd0.1`. The conclusion that SGD gives no speedup does not depend on this.

**Why `batch`.** With batch size 128, the measured speedup in `sweep` is consistently below 1/‖p‖₂ (9.5 vs 26.6 at T = 1, 1.41 vs 1.85 at T = 0.1). The blog's argument is per sample. A larger batch averages away part of the label noise in the regular gradient, which shrinks Adam's √v and makes regular training faster than `lr·‖p‖₂`. If that is the cause, the measured speedup should move toward 1/‖p‖₂ as B gets smaller. With B = 1 the curves are very noisy, so a single seed can be off, and a run that never reaches the threshold within 3000 steps is recorded as NaN.

Every figure is saved as a `.png` plus a `.json` file with the raw curves. Each figure also has an `*_eval.png` version, which plots the clean soft-CE loss against the teacher on a fixed held-out set of 8192 points. That curve has no label-sampling noise and puts both methods on the same scale. The main figures plot the training loss, as the blog does.

The measured speedup in `sweep` and `batch` is `steps_regular / steps_distill`, where each count is the number of steps the eval loss needs to close 50% of the gap between its initial value and the teacher entropy.

## Files

- `toy_distill.py`: teacher, student, training loop, and teacher statistics (entropy, ‖p‖₂).
- `run_experiments.py`: experiments and plotting.
