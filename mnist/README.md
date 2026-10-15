# MNIST distillation — reproduction

Reproduces the MNIST experiment (Section 3) of Hinton, Vinyals & Dean, [*Distilling the Knowledge in a Neural Network*](https://arxiv.org/abs/1503.02531) (2015).

| Model | Paper test errors |
|---|---|
| Teacher: 784-1200-1200-10, dropout + max-norm + jitter | 67 |
| Student: 784-800-800-10, no regularization, hard labels | 146 |
| Student: 784-800-800-10, distilled at T = 20 | 74 |

## Teacher

`train_teacher.py` trains the teacher. The paper says only that it is "strongly regularized using dropout and weight-constraints as described in [5]" and that inputs are "jittered by up to two pixels in any direction". It gives no training hyperparameters, so the defaults are those of its reference [5], Hinton et al. (2012), [*Improving neural networks by preventing co-adaptation of feature detectors*](https://arxiv.org/abs/1207.0580), Appendix A:

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

All of these can be changed with flags (`--help`). The whole dataset is kept as tensors on the device, so there is no DataLoader. MNIST is downloaded to `mnist/data/` on first run.

Run from the repository root:

```bash
uv sync
uv run mnist/train_teacher.py
```

The checkpoint goes to `mnist/checkpoints/teacher.pt` (`state_dict`, `config`, `history`), and the config and test-error curve go to `mnist/checkpoints/teacher.json`. A GPU is used when one is available. On the server GPU one epoch takes about 0.3 s, so 3000 epochs take roughly 15 minutes; on a 12-thread CPU an epoch takes about 10 s.

An earlier, shorter recipe (SGD lr 0.1, momentum 0.9, cosine decay, max-norm 3.5, PyTorch default init, 200 epochs) reached 108 test errors.

`common.py` holds the parts the student and distillation scripts will share: data loading, jitter, the `MLP` class, and test-error counting.
