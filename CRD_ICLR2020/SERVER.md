# CRD 复现：本地改代码，服务器运行

状态（2026-10-20）：已准备兼容性修改与运行脚本。未在本机安装依赖、运行 smoke test 或训练；服务器验证待完成。用户明确要求本机只修改代码。

## 目录与 Git

本实验位于 distillation/CRD_ICLR2020，由 distillation 父仓库统一管理，不再保留活动的嵌套 Git 仓库。原上游 commit 为 `b84f547c5db6a35318d4671d7d5c4de74c822403`，上游地址：https://github.com/HobbitLong/RepDistiller 。BSD-2-Clause LICENSE 保留。

原嵌套仓库元数据已移至父仓库 `.git/repdistiller-upstream.git`，原父仓库 ignore 文件备份为 `.git/gitignore-before-crd.backup`。这两项是本机备份，不会随 push 上传。尚未执行 git add、commit 或 push；把 CRD_ICLR2020 作为普通目录提交后，服务器才能通过父仓库获取修改。不要重新在 CRD_ICLR2020 内 git clone。

父目录 pyproject.toml、uv.lock 和已有 .venv 不变。此实验使用自己的 .venv，避免改变已有 MNIST 等实验环境。

## 服务器独立环境

以下操作全部在 Linux 训练服务器进行，先进入仓库的 CRD_ICLR2020 目录。建议 Python 3.11。这里固定一个已有官方 CUDA 12.8 wheel 的版本组合，不追随 latest：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-server.txt
nvidia-smi
python scripts/check_environment.py
```

安装来源：https://pytorch.org/get-started/previous-versions/#v271 。Blackwell 型号、服务器驱动、系统架构均尚未核实；不能把安装成功视为 GPU 可用。检查脚本打印实际 GPU、compute capability、编译架构，并进行 CUDA 卷积前向与反向。若失败，先保留报错，按具体 GPU/驱动调整安装组合。ARM 服务器需单独确认 wheel 支持。

可选：设置 `export CUDA_VISIBLE_DEVICES=0` 选择单张卡；不要更改 batch size 以求占满显存。多卡/DDP 不在本轮复现范围内。可设置 `export CRD_DATA_ROOT=/absolute/path/to/cifar100`；默认数据存入项目 data/，torchvision 首次会下载 CIFAR-100，需联网。

## 第一步：服务器 smoke test

```bash
python scripts/run_experiment.py teacher --smoke
python scripts/run_experiment.py vanilla --smoke
```

teacher smoke 生成的 checkpoint 为：

`runs/teacher_trial1_seed0_smoke/save/models/resnet32x4_cifar100_lr_0.05_decay_0.0005_trial_1_seed_0_smoke/resnet32x4_last.pth`

```bash
SMOKE_TEACHER="$PWD/runs/teacher_trial1_seed0_smoke/save/models/resnet32x4_cifar100_lr_0.05_decay_0.0005_trial_1_seed_0_smoke/resnet32x4_last.pth"
python scripts/run_experiment.py kd --smoke --teacher "$SMOKE_TEACHER"
python scripts/run_experiment.py crd --smoke --teacher "$SMOKE_TEACHER"
```

每项只跑 1 epoch、2 个训练 batch、2 个验证 batch，num_workers=0；CRD 仍使用完整 50000 样本 memory bank、16384 negatives，以检查正式配置的主要显存需求。会下载完整数据；smoke 不等于小数据训练。这个 teacher 只是接口检查用，不能用于正式蒸馏。脚本要求指标有限，并记录 smoke_passed；尚未在任何服务器执行。正式数据加载多进程也需在完整运行首轮观察。

## 第二步：正式 teacher 与 vanilla

先选一种 teacher 来源，KD 和 CRD 始终使用同一份 teacher：

- 训练 teacher：执行下面命令，记录它的完整验证精度。
- 使用上游发布的 teacher：参阅原 README 和下载脚本，确认来源。用 --teacher 指定 checkpoint，架构按配置为 resnet32x4。若旧文件含 argparse.Namespace 等对象，只有信任来源时才加 --trusted-checkpoint；默认采用 weights_only=True。不要把随机初始化或 smoke teacher 当作正式 teacher。

```bash
python scripts/run_experiment.py teacher --seed 0
python scripts/run_experiment.py vanilla --seed 0
```

正式 teacher 的默认路径：

```bash
TEACHER="$PWD/runs/teacher_trial1_seed0/save/models/resnet32x4_cifar100_lr_0.05_decay_0.0005_trial_1_seed_0/resnet32x4_last.pth"
```

## 第三步：KD、CRD，以及可选 CRD+KD

```bash
python scripts/run_experiment.py kd --seed 0 --teacher "$TEACHER"
python scripts/run_experiment.py crd --seed 0 --teacher "$TEACHER"
python scripts/run_experiment.py crd_kd --seed 0 --teacher "$TEACHER"
```

configs/cifar100.json 保存正式配置：240 epochs、batch 64、SGD lr 0.05、momentum 0.9、weight decay 5e-4；沿用上游在 epoch >150、>180、>210 衰减的实现。KD 的 CE/KL/CRD 权重是 0.1/0.9/0，CRD 是 1/0/0.8，CRD+KD 是 1/1/0.8；KL T=4，CRD feature_dim=128、K=16384、T=0.07、momentum=0.5、exact positives。权重来自上游 scripts/run_cifar_distill.sh。

先完成 seed 0，再对 vanilla/KD/CRD 使用 seed 1、2；保持同一 teacher 并记录其 SHA256。每项报告最后一轮 test top-1，以及三次运行均值和标准差；best 精度单列。上游说明论文/README 使用最后一轮结果，不能拿本地 best 与其比较。上游 resnet32x4→resnet8x4 参考值：teacher 79.42%、vanilla 72.50%、KD 73.33%、CRD 75.51%；这些是上游基准，不是本次实测，现代软件栈和重新训练 teacher 可能产生差异。

## 记录、失败与重复运行

每个 runs/<stage>_trial<N>_seed<S>[_smoke]/ 下保存 manifest.json（命令、配置、teacher SHA256、退出状态、最后一轮、best）、pip-freeze.txt、console.log，以及 save/ 下的 config.json、environment.json、metrics.jsonl、working-tree.patch 和权重。TensorBoard 也在该次运行的 save/ 下。数据、日志和权重不进 Git；需要同步的结果摘要请写在 RESULTS.md。

同名运行目录存在就拒绝覆盖；重跑用 --trial 2。尚未实现中断恢复，失败后用新 trial 从头运行，不要把重新开始称为 resume。若改变配置文件，请使用 --config 指定另一个文件并选新 trial，保留原配置。

commit 后再同步到服务器，保证源码可追溯。working-tree.patch 不包含 Git 未跟踪文件，不能替代提交源码。脚本保存实际 pip freeze，但它不是跨平台 lockfile。

默认设置随机种子并关闭 TF32；不保证逐位一致。需要更严格复现时给每项加 --deterministic，脚本同时设置 CUBLAS_WORKSPACE_CONFIG；不支持确定性运算时会报错，勿悄悄关闭后继续混用结果。

## 修改范围

- torchvision CIFAR-100 的 train_data/train_labels 更新为 data/targets。
- tensorboard_logger 替换为 torch.utils.tensorboard，保留原训练调用方式。
- CRD alias sampler 按输入 device 移动，取消构造时强制 .cuda()。
- checkpoint 默认安全 weights-only 加载到 CPU，teacher 架构可显式指定；新 last checkpoint 中 opt 改为普通字典。
- KD 使用 reduction='sum'，保留原损失缩放；不改蒸馏目标。
- 新增 seed、smoke batch 上限、运行元数据、逐轮指标和防覆盖规则。

本次只针对 CIFAR-100 vanilla/KD/CRD 路径准备兼容性修改；其他蒸馏方法和 ImageNet 未验证。全部运行验证留待服务器完成。
