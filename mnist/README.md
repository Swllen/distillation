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
├── shift_bias.py        test errors when the output biases of digits left out of training are raised
├── reproduce.ipynb      runs every experiment below and plots the results
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
| `shift_bias.py` | `shift_errors`: adds the same value to the output biases of the given digits and counts test errors |

Each run follows the same path: settings → `make_model` → `train` → `save_run`.

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

To leave digits out of the transfer set, then raise the biases of the left-out digits on the trained model:

```bash
# no 3s (paper: 206 errors; 109 after raising the 3's bias by 3.5)
uv run mnist/train_student.py --config mnist/configs/student_distill.toml --exclude-digits 3 --name distill_no3 --out mnist/checkpoints/distill_no3.pt
uv run mnist/shift_bias.py mnist/checkpoints/distill_no3.pt

# only 7s and 8s (paper: 47.3% errors; 13.2% after lowering the biases of 7 and 8 by 7.6)
uv run mnist/train_student.py --config mnist/configs/student_distill.toml --exclude-digits 0 1 2 3 4 5 6 9 --name distill_only78 --out mnist/checkpoints/distill_only78.pt
uv run mnist/shift_bias.py mnist/checkpoints/distill_only78.pt
```

To run everything in a notebook instead:

```bash
uv run --with jupyter jupyter lab mnist/reproduce.ipynb
```
