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

Run from the repository root.

```bash
uv sync
uv run mnist/train_teacher.py --config mnist/configs/teacher.toml
uv run mnist/train_student.py --config mnist/configs/student_hard.toml
uv run mnist/train_student.py --config mnist/configs/student_distill.toml
```

Train the teacher first; the two students can run at the same time. A run with `configs/X.toml` saves `mnist/checkpoints/X.pt` and `X.json`. To change a setting, edit a copy of a config file, or pass it as a flag, e.g. `--seed 1`.

Runs log to Weights & Biases (project `distillation-mnist`) after `uv run wandb login`; pass `--no-wandb` to turn this off.

To run everything in a notebook instead:

```bash
uv run --with jupyter jupyter lab mnist/reproduce.ipynb
```
