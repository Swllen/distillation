"""Run metadata, W&B logging and bounded smoke runs, without changing full-run defaults."""
import argparse
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import wandb


def add_arguments(parser):
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_batches', type=int, default=0,
                        help='0: full epoch; positive: smoke test train/validation limit')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True,
                        help='log to Weights & Biases (default on; WANDB_MODE=offline also works)')
    parser.add_argument('--wandb_project', default='distillation-crd')
    parser.add_argument('--wandb_entity', default='', help='W&B user or team; default your own account')
    parser.add_argument('--wandb_name', default='', help='W&B run name; default the model name')
    parser.add_argument('--wandb_group', default='', help='W&B group, e.g. the stage (teacher/vanilla/kd/crd)')


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


def start_run(opt, job):
    """Start a W&B run (a no-op run with --no-wandb). Every metric is plotted against epoch."""
    run = wandb.init(
        project=opt.wandb_project,
        entity=opt.wandb_entity or None,
        name=opt.wandb_name or opt.model_name,
        group=opt.wandb_group or job,
        job_type=job,
        tags=['smoke'] if opt.max_batches else None,
        config=vars(opt),
        dir=opt.save_folder,
        mode=None if opt.wandb else 'disabled',  # None: online, or whatever WANDB_MODE says
    )
    run.define_metric('epoch')
    run.define_metric('*', step_metric='epoch')
    return run


def record_epoch(opt, epoch, train_acc, train_loss, test_acc, test_top5, test_loss, run=None, **extra):
    """Append one epoch to metrics.jsonl and log it to W&B; `extra` holds e.g. lr and loss terms."""
    values = dict(epoch=epoch, train_acc=float(train_acc), train_loss=float(train_loss),
                  test_acc=float(test_acc), test_top5=float(test_top5), test_loss=float(test_loss),
                  smoke=bool(opt.max_batches))
    values.update({key: float(value) for key, value in extra.items()})
    with open(Path(opt.save_folder) / 'metrics.jsonl', 'a', encoding='utf-8') as stream:
        stream.write(json.dumps(values) + '\n')
    if run is not None:
        run.log({'epoch': epoch, 'train/acc': values['train_acc'], 'train/loss': values['train_loss'],
                 'test/acc': values['test_acc'], 'test/top5': values['test_top5'],
                 'test/loss': values['test_loss'], **{'train/' + key: values[key] for key in extra}})
