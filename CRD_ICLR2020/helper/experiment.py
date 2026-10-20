"""Run metadata and bounded smoke runs, without changing full-run defaults."""
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def add_arguments(parser):
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_batches', type=int, default=0,
                        help='0: full epoch; positive: smoke test train/validation limit')
    parser.add_argument('--deterministic', action='store_true')


def prepare(opt):
    if opt.epochs < 1 or opt.batch_size < 2 or opt.max_batches < 0:
        raise ValueError('epochs >= 1, batch_size >= 2, max_batches >= 0 required')
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.benchmark = not opt.deterministic
    torch.backends.cudnn.deterministic = opt.deterministic
    torch.use_deterministic_algorithms(opt.deterministic)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    folder = Path(opt.save_folder)
    # Never overwrite an earlier run, including failed or interrupted runs.
    marker = folder / 'config.json'
    if marker.exists():
        raise FileExistsError(f'{folder} already contains a run; choose a new --trial')
    marker.write_text(json.dumps(vars(opt), indent=2), encoding='utf-8')
    try:
        root = Path(__file__).resolve().parents[1]
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        diff = subprocess.check_output(['git', 'diff', '--binary'], cwd=root)
        (folder / 'working-tree.patch').write_bytes(diff)
        status = subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True)
    except (OSError, subprocess.CalledProcessError):
        revision, status = 'unavailable', 'unavailable'
    environment = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                       cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                       git_revision=revision, git_status=status, command=sys.argv,
                       data_root=os.environ.get('CRD_DATA_ROOT', './data'))
    (folder / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')


def record_epoch(opt, epoch, train_acc, train_loss, test_acc, test_top5, test_loss):
    values = dict(epoch=epoch, train_acc=float(train_acc), train_loss=float(train_loss),
                  test_acc=float(test_acc), test_top5=float(test_top5), test_loss=float(test_loss),
                  smoke=bool(opt.max_batches))
    with open(Path(opt.save_folder) / 'metrics.jsonl', 'a', encoding='utf-8') as stream:
        stream.write(json.dumps(values) + '\n')
