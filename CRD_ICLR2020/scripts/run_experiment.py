"""Launch one isolated experiment and capture its configuration and console log."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['teacher', 'vanilla', 'kd', 'crd', 'crd_kd'])
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/cifar100.json')
    parser.add_argument('--teacher', type=Path)
    parser.add_argument('--trusted-checkpoint', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    distilled = args.stage in ['kd', 'crd', 'crd_kd']
    teacher = args.teacher.resolve() if args.teacher else None
    if distilled and (teacher is None or not teacher.is_file()):
        parser.error('KD/CRD require --teacher /absolute/path/to/checkpoint.pth')
    name = f'{args.stage}_trial{args.trial}_seed{args.seed}' + ('_smoke' if args.smoke else '')
    folder = ROOT / 'runs' / name
    folder.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-u', str(ROOT / ('train_student.py' if distilled else 'train_teacher.py'))]
    for key in ['batch_size', 'learning_rate', 'lr_decay_epochs', 'lr_decay_rate', 'weight_decay', 'momentum']:
        command += ['--' + key, str(config[key])]
    command += ['--epochs', str(1 if args.smoke else config['epochs']),
                '--num_workers', str(0 if args.smoke else config['num_workers']),
                '--seed', str(args.seed), '--trial', str(args.trial)]
    if args.smoke:
        command += ['--max_batches', '2', '--print_freq', '1']
    if args.deterministic:
        command += ['--deterministic']
    if distilled:
        command += ['--model_s', config['student'], '--model_t', config['teacher'],
                    '--path_t', str(teacher), '--distill', 'kd' if args.stage == 'kd' else 'crd']
        for key in ['kd_T', 'feat_dim', 'nce_k', 'nce_t', 'nce_m', 'mode']:
            command += ['--' + key, str(config[key])]
        for flag, value in zip(['-r', '-a', '-b'], config['loss_weights'][args.stage]):
            command += [flag, str(value)]
        if args.trusted_checkpoint:
            command += ['--trusted_checkpoint']
    else:
        command += ['--model', config['teacher' if args.stage == 'teacher' else 'student']]
    environment = os.environ.copy()
    environment['CRD_DATA_ROOT'] = str(Path(environment.get('CRD_DATA_ROOT', ROOT / 'data')).resolve())
    if args.deterministic:
        environment.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    manifest = dict(config=config, command=command, teacher=str(teacher) if teacher else None,
                    teacher_sha256=None, smoke=args.smoke, status='starting')
    if teacher:
        digest = hashlib.sha256()
        with teacher.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        manifest['teacher_sha256'] = digest.hexdigest()
    path = folder / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (folder / 'pip-freeze.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    code = 1
    try:
        with (folder / 'console.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=folder, env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace')
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if code == 0:
            metrics = list(folder.glob('save/**/metrics.jsonl'))
            if len(metrics) != 1:
                raise RuntimeError('Expected exactly one metrics file')
            rows = [json.loads(line) for line in metrics[0].read_text().splitlines()]
            if len(rows) != (1 if args.smoke else config['epochs']):
                raise RuntimeError('Incomplete epoch metrics')
            if not all(math.isfinite(row[key]) for row in rows for key in
                       ['train_acc', 'train_loss', 'test_acc', 'test_top5', 'test_loss']):
                raise RuntimeError('Non-finite metrics')
            manifest['last_epoch'] = rows[-1]
            manifest['best_test_acc'] = max(row['test_acc'] for row in rows)
            manifest['status'] = 'smoke_passed' if args.smoke else 'completed'
        else:
            manifest['status'] = 'failed'
    except BaseException:
        code = 1
        manifest['status'] = 'failed_or_interrupted'
        raise
    finally:
        manifest['exit_code'] = code
        path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
