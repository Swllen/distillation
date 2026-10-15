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

```bash
pip install -r requirements.txt
python run_experiments.py all            # every experiment, outputs in results/
python run_experiments.py temp speedup   # only some of them
python run_experiments.py all --batch-size 64 --lr 3e-3 --out results_bs64
```

The script uses CUDA when it is available and the CPU otherwise. The model is tiny, so the CPU works too.

| Experiment | What it does | Output |
|---|---|---|
| `temp` | Adam, T ∈ {1, 0.1, 0.01}, 1000 steps (blog fig. 1) | `results/temp.png` |
| `speedup` | Adam, T = 0.1, 3000 steps, log x-axis, raw and with the distillation x-axis rescaled by 1/‖p‖₂ (blog fig. 2) | `results/speedup.png` |
| `sgd` | SGD, T ∈ {1, 0.1, 0.01}, 3000 steps (blog fig. 3) | `results/sgd_temp.png` |
| `sweep` | Adam over 9 temperatures: measured burn-in speedup vs predicted 1/‖p‖₂ | `results/sweep.png` |

Every figure is saved as a `.png` plus a `.json` file with the raw curves. Each figure also has an `*_eval.png` version, which plots the clean soft-CE loss against the teacher on a fixed held-out set of 8192 points. That curve has no label-sampling noise and puts both methods on the same scale. The main figures plot the training loss, as the blog does.

The measured speedup in `sweep` is `steps_regular / steps_distill`, where each count is the number of steps the eval loss needs to close 50% of the gap between its initial value and the teacher entropy.

## Files

- `toy_distill.py`: teacher, student, training loop, and teacher statistics (entropy, ‖p‖₂).
- `run_experiments.py`: experiments and plotting.
