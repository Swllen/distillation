# MNIST distillation — reproduction

Reproduces the MNIST experiment (Section 3) of Hinton, Vinyals & Dean, [*Distilling the Knowledge in a Neural Network*](https://arxiv.org/abs/1503.02531) (2015).

| Model | Paper test errors |
|---|---|
| Teacher: 784-1200-1200-10, dropout + max-norm + jitter | 67 |
| Student: 784-800-800-10, no regularization, hard labels | 146 |
| Student: 784-800-800-10, distilled at T = 20 | 74 |

## Teacher

`train_teacher.py` trains the teacher. The paper says only that it is "strongly regularized using dropout and weight-constraints as described in [5]" and that inputs are "jittered by up to two pixels in any direction". It gives no training hyperparameters, so the settings in `configs/teacher.toml` are those of its reference [5], Hinton et al. (2012), [*Improving neural networks by preventing co-adaptation of feature detectors*](https://arxiv.org/abs/1207.0580), Appendix A:

| Setting | Value | Source |
|---|---|---|
| Architecture | 784-1200-1200-10, ReLU | paper |
| Jitter | each training image shifted by (dy, dx) ∈ {-2, …, 2}², zero-filled | paper |
| Dropout | 0.2 on input pixels, 0.5 on every hidden unit | [5] |
| Max-norm | incoming weight vector of each hidden unit: squared length ≤ 15 (L2 norm ≤ √15), rescaled after every update | [5] |
| Loss, batch | cross-entropy averaged over minibatches of 100 | [5] |
| Update | Δwₜ = pₜ Δwₜ₋₁ − (1 − pₜ) εₜ ⟨∇L⟩ | [5] |
| Learning rate | εₜ = 10 × 0.998ᵗ, t = epoch | [5] |
| Momentum | pₜ rises linearly 0.5 → 0.99 over the first 500 epochs, then 0.99 | [5] |
| Epochs | 3000 | [5] |
| Init | weights ~ N(0, 0.01²), biases 0 (bias init not stated) | [5] |

In PyTorch's momentum convention this is a learning rate of (1 − pₜ) εₜ: 5.0 at the start, 0.037 at epoch 500 and 0.0002 at the end. Not stated in either paper: the bias init and the jitter distribution (uniform over integer shifts, resampled every epoch).

The whole dataset is kept as tensors on the device, so there is no DataLoader. MNIST is downloaded to `mnist/data/` on first run.

### Results so far

| Run | Test errors |
|---|---|
| [5] recipe as written (ε₀ = 10, with biases) | never trains: ~9000 errors, loss stays at ln 10 from epoch 1 |
| same with ε₀ = 1, 50 epochs | 249 |
| **ε₀ = 1, max-norm 2, 3000 epochs** (everything else as [5]) | **66** (paper: 67); 62–66 over the last 400 epochs, 13 min on GPU |

The teacher used for distillation is the last row: [5]'s recipe with two changes, ε₀ = 1.0 instead of 10 and an L2 max-norm of 2 instead of √15.

With ε₀ = 10 the hidden layers stop contributing within the first steps: runs with `init_std` 0.01 and 0.1 give bit-identical results, so the output depends only on the output biases. The widely used reimplementation [mdenil/dropout](https://github.com/mdenil/dropout) follows [5] except that it uses ε₀ = 1.0 and no biases.

### Config files

All settings live in a TOML file, [`configs/teacher.toml`](configs/teacher.toml), which says what each one means and where its value comes from. To try other settings, copy the file under a new name and edit the copy, so that `git pull` never conflicts with your edits. The checkpoint is named after the config file, so runs do not overwrite each other. Any setting can also be overridden on the command line, and flags take precedence over the file. Run from the repository root:

```bash
uv sync
uv run mnist/train_teacher.py                                        # configs/teacher.toml
uv run mnist/train_teacher.py --config mnist/configs/my_run.toml     # your own copy
uv run mnist/train_teacher.py --lr 1 --no-bias --epochs 50           # quick override
```

A run with `configs/X.toml` writes `mnist/checkpoints/X.pt` (`state_dict`, `config`, `history`) and `mnist/checkpoints/X.json` (config and test-error curve); `--out` changes the path. The resolved config is printed at the start of every run. A GPU is used when one is available. On the server GPU one epoch takes about 0.3 s, so 3000 epochs take roughly 15 minutes; on a 12-thread CPU an epoch takes about 10 s.

An earlier, shorter recipe (SGD lr 0.1, momentum 0.9, cosine decay, max-norm 3.5, PyTorch default init, 200 epochs) reached 108 test errors.

## Students

`train_student.py` trains the 784-800-800-10 student in two ways, each with its own config:

| Config | Targets | Paper test errors |
|---|---|---|
| [`configs/student_hard.toml`](configs/student_hard.toml) | true labels only | 146 |
| [`configs/student_distill.toml`](configs/student_distill.toml) | teacher soft targets at T = 20, plus true labels | 74 |

Distillation loss (paper Section 2), with student logits z and teacher logits v on the same image:

  loss = (1 − hard_weight) · T² · CE(softmax(v/T), softmax(z/T)) + hard_weight · CE(y, softmax(z))

The T² factor keeps the soft-target gradients on the same scale as T changes. The student is evaluated at T = 1. The teacher runs in eval mode (no dropout) on the same minibatch as the student.

| Setting | Value | Source |
|---|---|---|
| Architecture | 784-800-800-10, ReLU | paper |
| Regularization | none: no dropout, no max-norm, no jitter | paper ("no regularization") |
| Temperature | 20 | paper |
| hard_weight | 0.1 | not given for MNIST; Section 2 says the hard term should get "a considerably lower weight" |
| Transfer set | the 60,000 training images, unjittered | not stated; Section 2 says the original training set works well |
| Optimizer, schedule, init, epochs | same as the 66-error teacher run (ε₀ = 1, 3000 epochs) | not given for the student |

The student runs have no max-norm, and ε₀ = 1 is too large for distillation: the distilled student died at once (loss 828.5 ≈ 0.9 · T² · ln 10, i.e. uniform outputs; 9026 test errors). With T² scaling the soft-target gradient on the logits is roughly (z − v)/10, so it grows with the size of the teacher's logits and can be several times larger than a hard-label gradient. With ε₀ = 0.1 the distilled student reached 96 test errors after 25 epochs. Use the same `lr` for both students so that only the targets differ:

```bash
uv run mnist/train_student.py --config mnist/configs/student_hard.toml --lr 0.1
uv run mnist/train_student.py --config mnist/configs/student_distill.toml --lr 0.1
```

The distillation loss contains the teacher's entropy H(softmax(v/T)), which the student cannot reduce, so the printed loss stays near 0.9 · 400 · H ≈ 777. W&B logs the part that does go to 0 as `train/soft_kl`.

`student_distill.toml` reads the teacher from `mnist/checkpoints/teacher.pt` and prints its test errors before training starts.

## Optimizer

Both scripts use the momentum update of [5] by default. Set `optimizer = "adam"` in a config file (or pass `--optimizer adam`) to use `torch.optim.Adam` instead:

| Setting | `momentum` (default) | `adam` |
|---|---|---|
| Update | Δw = pΔw − (1 − p) ε g | Adam with betas (`adam_beta1`, `adam_beta2`), default (0.9, 0.999) |
| `lr` | ε₀; 1 for the teacher, 0.1 for the students | Adam's learning rate, typically around 1e-3; the momentum values would diverge |
| `lr_decay` | ε multiplied by it every epoch | same |
| `momentum_initial/final/epochs` | momentum ramp | ignored |
| max-norm, dropout, jitter | applied | applied |

For example, add these lines to a copy of a config:

```toml
optimizer = "adam"
lr = 1e-3
```

These settings are optional, so config files without them keep using the momentum update.

### Learning-rate schedule

| `lr_schedule` | Learning rate at epoch t (from 0) | Uses |
|---|---|---|
| `"exponential"` (default, [5]) | lr · lr_decay^t | `lr`, `lr_decay` |
| `"cosine"` (cosine annealing) | lr_min + (lr − lr_min) · (1 + cos(π t / epochs)) / 2 | `lr`, `lr_min` (default 0), `epochs` |

The rate changes once per epoch and works with either optimizer. For example, Adam with cosine annealing from 1e-3 to 0:

```toml
optimizer = "adam"
lr = 1e-3
lr_schedule = "cosine"
```

## Dashboard (Weights & Biases)

Every run of `train_teacher.py` and `train_student.py` logs to [Weights & Biases](https://wandb.ai), project `distillation-mnist`. The run is named after the config file (`--name` changes it), and its full config, including command-line overrides, is stored with it. One-time setup on each machine:

```bash
uv sync
uv run wandb login        # paste the API key from https://wandb.ai/authorize
```

| Panel | Metrics | Logged |
|---|---|---|
| `train/` | `loss`, `grad_norm`; for distillation also `soft_kl`, `soft_ce`, `teacher_entropy`, `hard_ce` | mean over every `log_every` steps (default 100) |
| `optim/` | `lr` (ε), `effective_lr` ((1 − p) ε, the PyTorch-equivalent rate), `momentum` | every `log_every` steps |
| `test/` | `errors`, `loss`, `dead_units_fc1`, `dead_units_fc2` (hidden units never active on the test set) | every `eval_every` epochs |
| `train/` (eval) | `errors`, `eval_loss` on the 60,000 training images without dropout or jitter, `epoch_loss` | every `eval_every` epochs |
| `weights/` | `fc{1,2}_max_row_norm`, `fc{1,2}_frac_at_max_norm` (hidden units at the max-norm cap) | every `eval_every` epochs |

All panels use `epoch` as the x-axis. The run summary holds `final_test_errors`, `final_train_errors`, `paper_test_errors` (67, 146 or 74) and, for distillation, `teacher_test_errors`.

- No logging: `--no-wandb`. No network on the server: `WANDB_MODE=offline uv run ...`, then `uv run wandb sync wandb/offline-run-*` from a machine that has network.
- Sharing: a project is private to your account by default. Either make the project public or invite the viewer under the project's settings on wandb.ai, or put the panels in a W&B Report and share its link.
- `wandb/` (local run files) is in `.gitignore`.

## Notebook

[`reproduce.ipynb`](reproduce.ipynb) runs the whole experiment from top to bottom: the teacher, both students, a results table against the paper, and plots of the test errors, learning rate, momentum, training loss and soft-target KL. All settings are Python dicts in one cell, with a table of where each value comes from. The notebook calls the same code as the scripts (`train_teacher.run_teacher`, `train_student.run_student`); it does not need config files or W&B.

```bash
uv sync
uv run --with jupyter jupyter lab mnist/reproduce.ipynb
```

- `QUICK = True` runs 50 epochs per model to check the pipeline in a few minutes.
- `REUSE = True` skips a run whose saved results match its settings, so the plots can be redrawn without retraining.
- Checkpoints are named `checkpoints/nb_*.pt` (`nb_quick_*` in quick mode), so the notebook never overwrites the command-line runs.

## Code

`common.py` holds everything the two scripts share: data loading, jitter, the `MLP` class, config parsing, the optimizer of [5], the training loop with W&B logging, evaluation, and saving and loading checkpoints.
