# MNIST distillation — reproduction

Reproduces the MNIST experiment (Section 3) of Hinton, Vinyals & Dean, [*Distilling the Knowledge in a Neural Network*](https://arxiv.org/abs/1503.02531) (2015).

| Model | Paper test errors |
|---|---|
| Teacher: 784-1200-1200-10, dropout + max-norm + jitter | 67 |
| Student: 784-800-800-10, no regularization, hard labels | 146 |
| Student: 784-800-800-10, distilled at T = 20 | 74 |

## Teacher

`train_teacher.py` trains the teacher. The paper says only that it is "strongly regularized using dropout and weight-constraints" and that inputs are "jittered by up to two pixels in any direction". The optimizer settings are not given, so the defaults follow the dropout recipe of Srivastava et al. (2014):

- Dropout 0.2 on the input pixels, 0.5 on both hidden layers.
- Max-norm: each hidden unit's incoming weight vector is rescaled to L2 norm ≤ 3.5 after every step.
- Jitter: every training image is shifted by a random (dy, dx) ∈ [-2, 2]², zero-filled.
- SGD, momentum 0.9, batch 100, lr 0.1 with cosine decay to 0, 200 epochs.

All of these can be changed with flags (`--help`). The whole dataset is kept as tensors on the device, so there is no DataLoader. MNIST is downloaded to `mnist/data/` on first run.

Run from the repository root:

```bash
uv sync
uv run mnist/train_teacher.py
```

The checkpoint goes to `mnist/checkpoints/teacher.pt` (`state_dict`, `config`, `history`), and the config and test-error curve go to `mnist/checkpoints/teacher.json`. On a 12-thread CPU one epoch takes about 10 s, so the default 200 epochs take about 35 minutes. A GPU is used when one is available.

`common.py` holds the parts the student and distillation scripts will share: data loading, jitter, the `MLP` class, and test-error counting.
