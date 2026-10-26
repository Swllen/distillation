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

## 第四步（可选）：论文 Table 1 的其他蒸馏方法

stage 名与上游 `--distill` 的对应：`fitnet`→hint、`at`→attention、`sp`→similarity、`cc`→correlation、`vid`、`rkd`、`pkt`、`ab`→abound、`ft`→factor、`fsp`、`nst`。权重在 configs/cifar100.json 的 loss_weights 中，均取自上游 scripts/run_cifar_distill.sh：γ=1、α=0，β 分别为 FitNet 100（hint_layer 2）、AT 1000、SP 3000、CC 0.02、VID 1、RKD 1、PKT 30000、AB 1、FT 200、FSP 50、NST 50。AB、FT、FSP 先预训练 init_epochs=30 轮（connector / paraphraser / 学生），再做 240 轮训练；其中 AB、FSP 第二阶段只有交叉熵（loss_kd 记为 0），与上游一致。

用法与 KD/CRD 相同，例如：

```bash
uv run CRD_ICLR2020/scripts/run_experiment.py fitnet --smoke --teacher "$TEACHER" --trusted-checkpoint
uv run CRD_ICLR2020/scripts/run_experiment.py fitnet --seed 0 --teacher "$TEACHER" --trusted-checkpoint
```

这些方法只在本机 CPU 上用假数据和随机 teacher 跑通过 smoke 路径（每个方法 2 个 batch，两阶段方法 1 轮预训练），尚未在服务器和真实数据上运行；每个方法正式运行前先跑一次 smoke。CC 的损失在假数据上数值很大（~10⁹），真实数据上需观察是否发散。上游 resnet32x4→resnet8x4 参考值：FitNet 73.50、AT 73.44、SP 72.94、CC 72.97、VID 73.09、RKD 71.90、PKT 73.64、AB 73.17、FT 72.86、FSP 72.62、NST 73.30。

## OPAL（分支 crd-opal，只用于 KD/CRD）

[OPAL](https://github.com/xingjian-bai/opal)（0.3.8，[博客](https://xingjianbai.com/reactive-optimizers/)）包装原来的 SGD：训练中一个代理（Claude Code）定期读训练统计，改写优化器程序（学习率、momentum、更新规则等）。代理只看到训练 loss、参数名与形状、自己的分析结果和训练脚本的只读副本；每个 epoch 的测试精度只用 `opt.log` 记下，代理看不到。

加 `--opal` 时，学习率改由 `MultiStepLR` 每个 update 走一步（衰减点 150/180/210 × 每 epoch 的 update 数），与 adjust_learning_rate 给出的学习率相同；`helper/opal_wrap.py` 负责包装，`train_distill` 调用 `optimizer.step(loss=loss)`。本机用假数据验证过：KD、CRD 在 `--opal-backend keep`（不调用代理）下与不加 `--opal` 的每轮指标、学习率和最终权重逐位相同（含 OPAL 的第 50 步检查）。

服务器准备（一次）：Claude Code ≥ 2.1.280，用 Team 账户 `claude auth login`，`claude auth status --text` 显示 claude.ai 登录；环境中不能有 `ANTHROPIC_API_KEY`（会盖过 Team 登录）；`bwrap`、`socat` 可用（代理的 shell 沙箱）。

```bash
git fetch && git checkout crd-opal && uv sync
uv run opal doctor          # 免费、离线
uv run opal doctor --live   # 一次很小的代理调用
TEACHER="$PWD/CRD_ICLR2020/save/models/resnet32x4_vanilla/ckpt_epoch_240.pth"
uv run CRD_ICLR2020/scripts/run_experiment.py kd --smoke --teacher "$TEACHER" --trusted-checkpoint --opal --opal-backend keep
uv run CRD_ICLR2020/scripts/run_experiment.py kd --seed 0 --teacher "$TEACHER" --trusted-checkpoint --opal --opal-max-cost-usd 15
uv run CRD_ICLR2020/scripts/run_experiment.py crd --seed 0 --teacher "$TEACHER" --trusted-checkpoint --opal --opal-max-cost-usd 15
```

代理 shell 的沙箱在 OPAL 的 `agent-tmp/` 里建 Unix socket（`srt-mux-<pid>-<n>.sock`），Linux 限制这种路径最长 107 个字符，而 OPAL 的记录目录很深（约 180 个字符）。所以 `helper/opal_wrap.py` 在 OPAL 启动前把 `agent-tmp` 建成链接，指向 `/tmp/claude-<uid>/opal-XXXXXXXX`（Claude Code 自己的私有临时目录，权限 700，重启时清空）；OPAL 0.3.8 用 `.resolve()` 取这个路径，记录仍留在运行目录里。`opal doctor` 用的是 `/tmp`，查不出这个问题。2026-10-26 的 `crd_opal_trial2_seed0` 还没有这个修复：代理的 shell 全程不可用，也没能读到训练脚本，其结果（最后一轮 75.05）不能代表 OPAL 正常工作时的表现。

运行目录是 `runs/<stage>_opal_trial<N>_seed<S>/`，W&B 组 `<stage>_opal`；OPAL 自己的记录在其下 `save/student_model/*_opal/opal/`：`uv run opal show <该目录>` 每次决策一行，`uv run opal report <该目录>` 生成 report.html（损失、学习率、每次决策的理由和代码改动），`programs/vNNN` 保存每一版优化器代码。默认训练等待代理（`--opal-blocking false` 则不等）；决策次数默认由 OPAL 定（≤50，等待时间约不超过训练时间 4 倍），`--opal-decisions` 可改。Team 账户按席位额度计，额度用完时代理不可用，每次决策最多等 30 分钟（`--opal_unavailable_timeout`，train_student.py），之后跳过该决策、用当前程序继续训练。订阅登录下 `max_cost_usd` 依据 Claude Code 报告的估计花费，不是账单；第一次运行后用 `opal show` 确认有花费记录，否则只能靠决策次数限制用量。

代理写的优化器代码在训练进程中以你的权限运行，不在沙箱内；作者指出短、已调好的配方提升有限，且同设置重跑结果可能不同。notebook 第 4 节把 OPAL 运行单列为 `kd+opal`、`crd+opal`，plot_methods.py 的图不含它们。

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
- run_experiment.py 和 configs/cifar100.json 扩展到 Table 1 的其他蒸馏方法（第四步）；KD/CRD 生成的训练命令不变。helper/pretrain.py 的预训练阶段遵守 smoke 的 batch 上限，预训练损失记到 W&B 的 init/train_loss（epoch 1−init_epochs … 0）。

vanilla/KD/CRD 已在服务器 CIFAR-100 上完成 seed 0（见 RESULTS.md）；其他蒸馏方法只在本机做过假数据 smoke；ImageNet 未验证。
