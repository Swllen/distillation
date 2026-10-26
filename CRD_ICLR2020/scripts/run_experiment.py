"""Launch one isolated experiment and capture its configuration and console log.

Used from the command line (see SERVER.md) and by reproduce.ipynb through run_stage.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Distilled stage -> train_student.py --distill; their loss weights are config['loss_weights'][stage].
DISTILL = {'kd': 'kd', 'crd': 'crd', 'crd_kd': 'crd', 'fitnet': 'hint', 'at': 'attention', 'sp': 'similarity',
           'cc': 'correlation', 'vid': 'vid', 'rkd': 'rkd', 'pkt': 'pkt', 'ab': 'abound', 'ft': 'factor',
           'fsp': 'fsp', 'nst': 'nst'}
DISTILLED = list(DISTILL)
STAGES = ['teacher', 'vanilla'] + DISTILLED
# Two-stage methods: init_epochs of pretraining (helper/pretrain.py) before distillation.
TWO_STAGE = ['ab', 'ft', 'fsp']
# Lines shown with echo=False: one summary per epoch plus the start and end of a run.
SUMMARY = ('epoch ', ' * Acc', 'teacher accuracy', 'best accuracy', '==> loading', 'Files already', 'Downloading',
           '[opal]')


def run_name(stage, seed=0, trial=1, smoke=False, opal=False):
    return f'{stage}' + ('_opal' if opal else '') + f'_trial{trial}_seed{seed}' + ('_smoke' if smoke else '')


def build_command(stage, config, teacher, seed, trial, smoke, deterministic, trusted_checkpoint,
                  wandb, wandb_project, wandb_entity, opal=None):
    """opal: None, or dict(backend=, max_cost_usd=, decisions=, blocking=) for train_student.py --opal."""
    distilled = stage in DISTILLED
    name = run_name(stage, seed, trial, smoke, bool(opal))
    command = [sys.executable, '-u', str(ROOT / ('train_student.py' if distilled else 'train_teacher.py'))]
    for key in ['batch_size', 'learning_rate', 'lr_decay_epochs', 'lr_decay_rate', 'weight_decay', 'momentum']:
        command += ['--' + key, str(config[key])]
    command += ['--epochs', str(1 if smoke else config['epochs']),
                '--num_workers', str(0 if smoke else config['num_workers']),
                '--seed', str(seed), '--trial', str(trial)]
    if smoke:
        command += ['--max_batches', '2', '--print_freq', '1']
    if deterministic:
        command += ['--deterministic']
    command += ['--wandb' if wandb else '--no-wandb', '--wandb_project', wandb_project,
                '--wandb_name', name,
                '--wandb_group', stage + ('_opal' if opal else '') + ('_smoke' if smoke else '')]
    if wandb_entity:
        command += ['--wandb_entity', wandb_entity]
    if distilled:
        command += ['--model_s', config['student'], '--model_t', config['teacher'],
                    '--path_t', str(teacher), '--distill', DISTILL[stage]]
        for key in ['kd_T', 'feat_dim', 'nce_k', 'nce_t', 'nce_m', 'mode']:
            command += ['--' + key, str(config[key])]
        if stage == 'fitnet':
            command += ['--hint_layer', str(config['hint_layer'])]
        if stage in TWO_STAGE:
            command += ['--init_epochs', str(1 if smoke else config['init_epochs'])]
        for flag, value in zip(['-r', '-a', '-b'], config['loss_weights'][stage]):
            command += [flag, str(value)]
        if trusted_checkpoint:
            command += ['--trusted_checkpoint']
        if opal:
            command += ['--opal', '--opal_backend', opal['backend'],
                        '--opal_blocking', 'true' if opal['blocking'] else 'false']
            if opal.get('max_cost_usd') is not None:
                command += ['--opal_max_cost_usd', str(opal['max_cost_usd'])]
            if opal.get('decisions') is not None:
                command += ['--opal_decisions', str(opal['decisions'])]
    else:
        command += ['--model', config['teacher' if stage == 'teacher' else 'student']]
    return command


def stage_settings(config, stage):
    """The part of config that affects `stage`: two runs of a stage with equal settings are the same run."""
    settings = {key: value for key, value in config.items()
                if key not in ('loss_weights', 'hint_layer', 'init_epochs')}
    if stage in DISTILLED:
        settings['loss_weights'] = config['loss_weights'][stage]
    if stage == 'fitnet':
        settings['hint_layer'] = config['hint_layer']
    if stage in TWO_STAGE:
        settings['init_epochs'] = config['init_epochs']
    return settings


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run_stage(stage, config=ROOT / 'configs/cifar100.json', teacher=None, seed=0, trial=1, smoke=False,
              deterministic=False, trusted_checkpoint=False, wandb=True, wandb_project='distillation-crd',
              wandb_entity='', echo=True, opal=None):
    """Train one stage in runs/<stage>[_opal]_trial<N>_seed<S>[_smoke]/ and return its manifest.

    Refuses to reuse an existing run folder. Raises RuntimeError if training fails.
    echo=False prints only the per-epoch summary lines; console.log always has everything.
    """
    if stage not in STAGES:
        raise ValueError(f'stage must be one of {STAGES}')
    config_path = Path(config)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    teacher = Path(teacher).resolve() if teacher else None
    if stage in DISTILLED and (teacher is None or not teacher.is_file()):
        raise ValueError('KD/CRD require a teacher checkpoint (--teacher /absolute/path/to/checkpoint.pth)')
    if opal and stage not in DISTILLED:
        raise ValueError('--opal is for distilled stages (train_student.py) only')
    name = run_name(stage, seed, trial, smoke, bool(opal))
    folder = ROOT / 'runs' / name
    folder.mkdir(parents=True, exist_ok=False)
    command = build_command(stage, config, teacher, seed, trial, smoke, deterministic, trusted_checkpoint,
                            wandb, wandb_project, wandb_entity, opal)
    environment = os.environ.copy()
    environment['CRD_DATA_ROOT'] = str(Path(environment.get('CRD_DATA_ROOT', ROOT / 'data')).resolve())
    if deterministic:
        environment.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    manifest = dict(stage=stage, seed=seed, trial=trial, config_file=str(config_path), config=config,
                    command=command, teacher=str(teacher) if teacher else None,
                    teacher_sha256=sha256(teacher) if teacher else None, smoke=smoke, opal=opal, status='starting')
    path = folder / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    # Installed packages, read directly so that it also works in uv venvs, which have no pip.
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    (folder / 'pip-freeze.txt').write_text('\n'.join(packages) + '\n', encoding='utf-8')
    code = 1
    try:
        with (folder / 'console.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=folder, env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace')
            try:
                for line in process.stdout:
                    if echo or line.startswith(SUMMARY):
                        print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if code == 0:
            rows = read_metrics(folder)
            if len(rows) != (1 if smoke else config['epochs']):
                raise RuntimeError('Incomplete epoch metrics')
            if not all(math.isfinite(row[key]) for row in rows for key in
                       ['train_acc', 'train_loss', 'test_acc', 'test_top5', 'test_loss']):
                raise RuntimeError('Non-finite metrics')
            manifest['last_epoch'] = rows[-1]
            manifest['best_test_acc'] = max(row['test_acc'] for row in rows)
            manifest['status'] = 'smoke_passed' if smoke else 'completed'
        else:
            manifest['status'] = 'failed'
    except BaseException:
        code = 1
        manifest['status'] = 'failed_or_interrupted'
        raise
    finally:
        manifest['exit_code'] = code
        path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if code != 0:
        if not echo:
            print(''.join((folder / 'console.log').read_text(encoding='utf-8').splitlines(True)[-30:]))
        raise RuntimeError(f'{name} failed (exit code {code}); see {folder / "console.log"}')
    return manifest


def read_metrics(folder):
    """The per-epoch rows of one run folder's metrics.jsonl."""
    # save/<models or student_model>/<model name>/metrics.jsonl only: OPAL writes another metrics.jsonl below it
    metrics = list(Path(folder).glob('save/*/*/metrics.jsonl'))
    if len(metrics) != 1:
        raise RuntimeError(f'Expected exactly one metrics file in {folder}')
    return [json.loads(line) for line in metrics[0].read_text(encoding='utf-8').splitlines()]


def checkpoint(folder):
    """The final (last-epoch) weights of one run folder."""
    files = list(Path(folder).glob('save/**/*_last.pth'))
    if len(files) != 1:
        raise RuntimeError(f'Expected exactly one *_last.pth in {folder}')
    return files[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/cifar100.json')
    parser.add_argument('--teacher', type=Path)
    parser.add_argument('--trusted-checkpoint', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True,
                        help='log to Weights & Biases (default on)')
    parser.add_argument('--wandb-project', default='distillation-crd')
    parser.add_argument('--wandb-entity', default='', help='W&B user or team; default your own account')
    parser.add_argument('--opal', action='store_true', help='OPAL: a coding agent rewrites the optimizer (KD/CRD...)')
    parser.add_argument('--opal-backend', default='claude', help='claude, or keep (no agent, no cost)')
    parser.add_argument('--opal-max-cost-usd', type=float, help='stop asking the agent after this estimated spend')
    parser.add_argument('--opal-decisions', type=int, help="times the agent is asked; default OPAL's")
    parser.add_argument('--opal-blocking', default='true', choices=['true', 'false'],
                        help='true: training waits for each decision')
    args = parser.parse_args()
    opal = dict(backend=args.opal_backend, max_cost_usd=args.opal_max_cost_usd, decisions=args.opal_decisions,
                blocking=args.opal_blocking == 'true') if args.opal else None
    try:
        run_stage(args.stage, args.config, args.teacher, args.seed, args.trial, args.smoke,
                  args.deterministic, args.trusted_checkpoint, args.wandb, args.wandb_project,
                  args.wandb_entity, opal=opal)
    except ValueError as error:
        parser.error(str(error))
    except RuntimeError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
