# MNIST distillation

Reproduces the MNIST experiment (Section 3) of Hinton, Vinyals & Dean, [*Distilling the Knowledge in a Neural Network*](https://arxiv.org/abs/1503.02531) (2015):

1. **Teacher**: a 784-1200-1200-10 ReLU net, regularized with dropout, a max-norm weight constraint and ±2-pixel input jitter.
2. **Student on true labels**: a 784-800-800-10 ReLU net with no regularization.
3. **Distilled student**: the same 800-800 net, trained on the teacher's soft targets at temperature T = 20 plus a low-weight true-label term.

The paper gives no training hyperparameters. The settings follow its reference [5], Hinton et al. (2012), [*Improving neural networks by preventing co-adaptation of feature detectors*](https://arxiv.org/abs/1207.0580), Appendix A, and each config file states where every value comes from.

## Code structure

```
mnist/
├── common.py            shared code: data, model, config parsing, optimizer, training loop, logging, checkpoints
├── train_teacher.py     trains the teacher
├── train_student.py     trains a student, on true labels or by distillation
├── reproduce.ipynb      runs all three trainings and plots the results
├── configs/
│   ├── teacher.toml
│   ├── student_hard.toml
│   └── student_distill.toml
├── data/                MNIST, downloaded on first run (not in git)
└── checkpoints/         <name>.pt and <name>.json per run (not in git)
```

| File | Contents |
|---|---|
| `common.py` | `load_mnist` (the whole dataset as tensors on the device), `jitter`, the `MLP` class with `apply_max_norm`, `parse_args` / `args_from_dict` (settings from a TOML file or a dict, with flags overriding), `make_model`, the learning-rate and momentum schedules, `train`, `evaluate`, `save_run` / `load_model`, and W&B setup (`start_run`) |
| `train_teacher.py` | `run_teacher(args)`: cross-entropy on true labels |
| `train_student.py` | `distillation_loss` and `run_student(args)`: true labels when `teacher = ""`, distillation otherwise |

Each run follows the same path: settings → `make_model` → `train` → `save_run`.

`train` does the following every step:
1. Jitters the minibatch.
2. Computes the loss terms. The script passes in the loss function.
3. Takes one optimizer step: either the momentum update of [5], Δw = pΔw − (1 − p)·ε·∇L, or Adam.
4. Projects each hidden unit's incoming weights onto the max-norm ball.

Every `eval_every` epochs, `train` evaluates on the training and test sets and logs the results.

For distillation, the teacher is loaded in eval mode and computes its logits on the same minibatch as the student. The loss is

  (1 − hard_weight) · T² · CE(softmax(v/T), softmax(z/T)) + hard_weight · CE(y, softmax(z))

where z are the student's logits and v the teacher's.

## Usage

Run everything from the repository root.

### Setup

```bash
uv sync
uv run wandb login          # optional: API key from https://wandb.ai/authorize
```

### Training

Train the teacher first. The two students are independent of each other and can run at the same time.

```bash
uv run mnist/train_teacher.py --config mnist/configs/teacher.toml
uv run mnist/train_student.py --config mnist/configs/student_hard.toml
uv run mnist/train_student.py --config mnist/configs/student_distill.toml
```

Each run prints its resolved settings first, then the test errors every `eval_every` epochs. On a GPU, a 3000-epoch run takes 10–20 minutes.

### Settings

To try other settings, copy a config file under a new name and edit the copy. The tracked files then stay unchanged and `git pull` does not conflict. Any setting can also be given as a flag, which takes precedence over the file:

```bash
uv run mnist/train_student.py --config mnist/configs/student_distill.toml --temperature 8 --hard-weight 0
```

These settings are optional and fall back to the defaults below:

| Setting | Default | Other values |
|---|---|---|
| `optimizer` | `"momentum"` (the update of [5]) | `"adam"` (`adam_beta1`, `adam_beta2`; use an `lr` around 1e-3) |
| `lr_schedule` | `"exponential"`: lr · lr_decay^epoch | `"cosine"`: from lr down to `lr_min` (default 0) |
| `wandb` | `true` | `false`, or `--no-wandb` |
| `wandb_project`, `wandb_entity`, `name` | `distillation-mnist`, your account, the config file name | |
| `log_every` | 100 | steps between W&B training logs |

### Outputs and naming

A run with `configs/X.toml` writes these files:
- `mnist/checkpoints/X.pt`: weights, settings and history;
- `mnist/checkpoints/X.json`: settings and history.

The W&B run is also named `X`. Running the same config again overwrites its files. To keep both, pass `--out` and `--name`, for example for another seed:

```bash
uv run mnist/train_student.py --config mnist/configs/student_distill.toml --seed 1 --name student_distill_s1 --out mnist/checkpoints/student_distill_s1.pt
```

Runs from the same config file share a W&B group, so W&B can show their mean and spread.

### Weights & Biases

Runs log to project `distillation-mnist`:

| Panel | Metrics |
|---|---|
| `train/` | loss terms and `grad_norm`; for distillation also `soft_kl`, `soft_ce`, `teacher_entropy` and `hard_ce` |
| `optim/` | `lr` and `momentum` |
| `test/` | errors, loss and the fraction of dead hidden units |
| `train/` (evaluation) | errors and loss on the training set |
| `weights/` | hidden-unit weight norms |

The x-axis is the epoch.

- Turn logging off with `--no-wandb`.
- On a machine without network, use `WANDB_MODE=offline uv run ...` and later run `uv run wandb sync wandb/offline-run-*`.

### Notebook

```bash
uv run --with jupyter jupyter lab mnist/reproduce.ipynb
```

`reproduce.ipynb` runs the three trainings from top to bottom with the settings as Python dicts, then shows a results table against the paper and plots of the training curves. It calls the same code as the scripts and does not log to W&B by default.

- `QUICK = True` runs 50 epochs per model, to check that everything works.
- `REUSE = True` skips runs whose saved results match their settings.
- Its checkpoints are named `checkpoints/nb_*.pt`.
