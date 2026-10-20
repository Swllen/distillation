# CRD 复现：本地改代码，服务器运行

状态（2026-10-20）：已准备兼容性修改、运行脚本、W&B 记录和 reproduce.ipynb。本机只在 CPU 上用假数据检查了 W&B 记录与损失分项；未下载 CIFAR-100、未跑 smoke test 或训练，服务器验证待完成。

## 两种运行方式

- **Notebook（给老师复现）**：`reproduce.ipynb` 从上到下运行即完成 teacher → vanilla → KD → CRD，并给出结果表和曲线。它只调用 `scripts/run_experiment.py` 里的 `run_stage`，与命令行训练的是同一套东西。顶部 `SMOKE = True` 时每项只跑 1 epoch、2 个 batch，用来先检查流程。
- **命令行**：下面各步的 `uv run CRD_ICLR2020/scripts/run_experiment.py ...`（在仓库根目录运行）。在 tmux 里跑完后，notebook（`REUSE = True`）直接读取已完成的 runs/，不重复训练。

两种方式都会记录到 Weights & Biases，项目 `distillation-crd`，每个 stage 一个 group（smoke 为 `<stage>_smoke`），run 名与 runs/ 下文件夹同名。记录：`train/acc`、`train/loss`、`train/lr`、`train/epoch_time`，学生另有未加权的 `train/loss_cls`、`train/loss_div`、`train/loss_kd`；`test/acc`、`test/top5`、`test/loss`；summary 中有 `teacher_acc`、`last_test_acc`、`best_test_acc`。命令行加 `--no-wandb` 关闭，notebook 设 `WANDB = False`；服务器不能联网时设 `export WANDB_MODE=offline`，之后用 `wandb sync` 上传。

## 目录与 Git

本实验位于 distillation/CRD_ICLR2020，由 distillation 父仓库统一管理，不再保留活动的嵌套 Git 仓库。原上游 commit 为 `b84f547c5db6a35318d4671d7d5c4de74c822403`，上游地址：https://github.com/HobbitLong/RepDistiller 。BSD-2-Clause LICENSE 保留。

原嵌套仓库元数据已移至父仓库 `.git/repdistiller-upstream.git`，原父仓库 ignore 文件备份为 `.git/gitignore-before-crd.backup`。这两项是本机备份，不会随 push 上传。CRD_ICLR2020 作为普通目录提交在父仓库中，服务器 clone/pull 父仓库即可获取修改。不要重新在 CRD_ICLR2020 内 git clone。

## 环境：与 MNIST 共用根目录的 uv 环境

CRD 和 mnist/、toy_distillation/ 一样，使用仓库根目录的 pyproject.toml、uv.lock 和 .venv，不另建环境；CRD 所需的 torch、torchvision、wandb、matplotlib 都已在其中。在 Linux 上 uv.lock 解析出带 CUDA 的 torch。以下命令都在服务器的仓库根目录 distillation/ 下运行，和 MNIST 相同：

```bash
uv sync
nvidia-smi
uv run CRD_ICLR2020/scripts/check_environment.py
uv run wandb login
```

检查脚本打印实际 GPU、compute capability、torch 编译支持的架构，并进行 CUDA 卷积前向与反向；安装成功不等于 GPU 可用。若失败，先保留报错再调整。

用 notebook 时，在 tmux 中启动 Jupyter（断开 SSH 不会中断训练），本地用端口转发打开：

```bash
tmux new -s crd
uv run --with jupyter jupyter lab --no-browser --port 8888 CRD_ICLR2020/reproduce.ipynb
# 本地另开终端：ssh -N -L 8888:localhost:8888 <user>@<server>，再在浏览器打开 Jupyter 打印的 http://localhost:8888/... 链接
```

可选：设置 `export CUDA_VISIBLE_DEVICES=0` 选择单张卡；不要更改 batch size 以求占满显存。多卡/DDP 不在本轮复现范围内。可设置 `export CRD_DATA_ROOT=/absolute/path/to/cifar100`；默认数据存入 CRD_ICLR2020/data/，torchvision 首次会下载 CIFAR-100，需联网。

## 第一步：服务器 smoke test

```bash
uv run CRD_ICLR2020/scripts/run_experiment.py teacher --smoke
uv run CRD_ICLR2020/scripts/run_experiment.py vanilla --smoke
```

teacher smoke 生成的 checkpoint 为：

`CRD_ICLR2020/runs/teacher_trial1_seed0_smoke/save/models/resnet32x4_cifar100_lr_0.05_decay_0.0005_trial_1_seed_0_smoke/resnet32x4_last.pth`

```bash
SMOKE_TEACHER="$PWD/CRD_ICLR2020/runs/teacher_trial1_seed0_smoke/save/models/resnet32x4_cifar100_lr_0.05_decay_0.0005_trial_1_seed_0_smoke/resnet32x4_last.pth"
uv run CRD_ICLR2020/scripts/run_experiment.py kd --smoke --teacher "$SMOKE_TEACHER"
uv run CRD_ICLR2020/scripts/run_experiment.py crd --smoke --teacher "$SMOKE_TEACHER"
```

每项只跑 1 epoch、2 个训练 batch、2 个验证 batch，num_workers=0；CRD 仍使用完整 50000 样本 memory bank、16384 negatives，以检查正式配置的主要显存需求。会下载完整数据；smoke 不等于小数据训练。这个 teacher 只是接口检查用，不能用于正式蒸馏。脚本要求指标有限，并记录 smoke_passed；尚未在任何服务器执行。正式数据加载多进程也需在完整运行首轮观察。

## 第二步：正式 teacher 与 vanilla

先选一种 teacher 来源，KD 和 CRD 始终使用同一份 teacher：

- 训练 teacher：执行下面命令，记录它的完整验证精度。
- 使用上游发布的 teacher：参阅原 README 和下载脚本，确认来源。用 --teacher 指定 checkpoint，架构按配置为 resnet32x4。若旧文件含 argparse.Namespace 等对象，只有信任来源时才加 --trusted-checkpoint；默认采用 weights_only=True。不要把随机初始化或 smoke teacher 当作正式 teacher。

```bash
uv run CRD_ICLR2020/scripts/run_experiment.py teacher --seed 0
uv run CRD_ICLR2020/scripts/run_experiment.py vanilla --seed 0
```

正式 teacher 的默认路径：

```bash
TEACHER="$PWD/CRD_ICLR2020/runs/teacher_trial1_seed0/save/models/resnet32x4_cifar100_lr_0.05_decay_0.0005_trial_1_seed_0/resnet32x4_last.pth"
```

## 第三步：KD、CRD，以及可选 CRD+KD

```bash
uv run CRD_ICLR2020/scripts/run_experiment.py kd --seed 0 --teacher "$TEACHER"
uv run CRD_ICLR2020/scripts/run_experiment.py crd --seed 0 --teacher "$TEACHER"
uv run CRD_ICLR2020/scripts/run_experiment.py crd_kd --seed 0 --teacher "$TEACHER"
```

configs/cifar100.json 保存正式配置：240 epochs、batch 64、SGD lr 0.05、momentum 0.9、weight decay 5e-4；沿用上游在 epoch >150、>180、>210 衰减的实现。KD 的 CE/KL/CRD 权重是 0.1/0.9/0，CRD 是 1/0/0.8，CRD+KD 是 1/1/0.8；KL T=4，CRD feature_dim=128、K=16384、T=0.07、momentum=0.5、exact positives。权重来自上游 scripts/run_cifar_distill.sh。

先完成 seed 0，再对 vanilla/KD/CRD 使用 seed 1、2；保持同一 teacher 并记录其 SHA256。每项报告最后一轮 test top-1，以及三次运行均值和标准差；best 精度单列。上游说明论文/README 使用最后一轮结果，不能拿本地 best 与其比较。上游 resnet32x4→resnet8x4 参考值：teacher 79.42%、vanilla 72.50%、KD 73.33%、CRD 75.51%；这些是上游基准，不是本次实测，现代软件栈和重新训练 teacher 可能产生差异。

## 记录、失败与重复运行

每个 runs/<stage>_trial<N>_seed<S>[_smoke]/ 下保存 manifest.json（命令、配置、teacher SHA256、退出状态、最后一轮、best）、pip-freeze.txt、console.log，以及 save/ 下的 config.json、environment.json、metrics.jsonl（含 lr 和学生的损失分项）、working-tree.patch、权重和 W&B 本地文件（wandb/）。数据、日志和权重不进 Git；需要同步的结果摘要请写在 RESULTS.md。

同名运行目录存在就拒绝覆盖；重跑用 --trial 2。尚未实现中断恢复，失败后用新 trial 从头运行，不要把重新开始称为 resume。若改变配置文件，请使用 --config 指定另一个文件并选新 trial，保留原配置。

commit 后再同步到服务器，保证源码可追溯。working-tree.patch 不包含 Git 未跟踪文件，不能替代提交源码。pip-freeze.txt 是脚本从 importlib.metadata 读出的实际已安装包版本（uv 环境没有 pip 也可用）；环境本身由根目录 uv.lock 固定。

默认设置随机种子并关闭 TF32；不保证逐位一致。需要更严格复现时给每项加 --deterministic，脚本同时设置 CUBLAS_WORKSPACE_CONFIG；不支持确定性运算时会报错，勿悄悄关闭后继续混用结果。

## 修改范围

- torchvision CIFAR-100 的 train_data/train_labels 更新为 data/targets。
- 去掉 tensorboard_logger / TensorBoard，训练曲线只记录到 W&B 和 metrics.jsonl。
- CRD alias sampler 按输入 device 移动，取消构造时强制 .cuda()。
- checkpoint 默认安全 weights-only 加载到 CPU，teacher 架构可显式指定；新 last checkpoint 中 opt 改为普通字典。
- KD 使用 reduction='sum'，保留原损失缩放；不改蒸馏目标。
- 新增 seed、smoke batch 上限、运行元数据、逐轮指标和防覆盖规则。
- 新增 W&B 记录（helper/experiment.py 的 start_run / record_epoch）；train_distill 额外返回未加权的 loss_cls / loss_div / loss_kd，训练本身不变。
- run_experiment.py 拆出 run_stage 供 reproduce.ipynb 调用；命令行用法不变，另加 --no-wandb、--wandb-project、--wandb-entity。

本次只针对 CIFAR-100 vanilla/KD/CRD 路径准备兼容性修改；其他蒸馏方法和 ImageNet 未验证。全部运行验证留待服务器完成。
