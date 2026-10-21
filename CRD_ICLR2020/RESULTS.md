# 实验记录

CIFAR-100，teacher resnet32x4 → student resnet8x4，配置 configs/cifar100.json（240 epochs）。论文报告最后一轮的 test top-1，这里同样以最后一轮为准；best 是在测试集上挑出的，只作参考。

## 2026-10-20：seed 0

环境：NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition，torch 2.14.1+cu130（根目录共享 uv 环境）。三个正式运行的源码均为提交 `2b579bc`，工作区无未提交修改（environment.json 中 git_status 为空）。

teacher 没有重新训练，用的是作者发布的 `resnet32x4_vanilla/ckpt_epoch_240.pth`（scripts/fetch_pretrained_teachers.sh 下载），SHA256 `22aa12e2b632b19ddde155ccf2388671842ddd304824b3095e0a7fff364beb4f`。KD、CRD 运行开始前在完整测试集上测得 teacher 79.42%，与论文一致。

| 方法 | 最后一轮 top-1 | 论文 | 差 | best top-1 | 最后一轮 top-5 | 最后一轮 train top-1 | 每 epoch 训练时间 |
|---|---|---|---|---|---|---|---|
| teacher（作者权重） | 79.42 | 79.42 | 0.00 | — | — | — | — |
| vanilla | 72.50 | 72.50 | 0.00 | 73.11 | 92.75 | 95.19 | 5.2 s |
| KD | 73.22 | 73.33 | −0.11 | 73.41 | 92.89 | 88.51 | 8.6 s |
| CRD | 75.90 | 75.51 | +0.39 | 76.02 | 93.93 | 92.34 | 15.0 s |

排序 CRD > KD > vanilla 与论文一致；单个 seed，±0.4 以内的差距在随机波动范围内。

运行目录（服务器 CRD_ICLR2020/runs/ 下），三个均为 status completed、exit_code 0、240 个 epoch、日志无 warning/NaN；学习率 1–150 epoch 0.05，151–180 0.005，181–210 0.0005，211–240 0.00005：

| 方法 | 运行目录 | teacher SHA256 |
|---|---|---|
| vanilla | `vanilla_trial1_seed0` | — |
| KD | `kd_trial1_seed0` | `22aa12e2…4beb4f` |
| CRD | `crd_trial1_seed0` | `22aa12e2…4beb4f` |

smoke（vanilla、KD、CRD，seed 0，trial 1）在正式运行前均已通过；结果不计入上表。teacher 未自行训练，因此没有 teacher smoke 和 teacher 运行。

## 待做

- seeds 1、2 的 vanilla、KD、CRD（同一 teacher），统计三次的均值与标准差。
