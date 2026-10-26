"""Wrap a student's optimizer with OPAL (https://github.com/xingjian-bai/opal): a coding agent rewrites the
optimizer while the model trains. Only used with train_student.py --opal."""
import os
import tempfile

import torch

# Linux limits a Unix socket path to 107 characters; the agent's sandbox makes sockets such as
# srt-mux-<pid>-<n>.sock (up to about 30 characters) in OPAL's agent-tmp folder.
SOCKET_PATH_MAX = 107
SOCKET_NAME_MAX = 30


def short_agent_tmp(run_dir):
    """Make <run_dir>/agent-tmp a link to a new private folder under /tmp/claude-<uid> (Claude Code's own temp
    folder), so the sandbox's sockets get a short path; the records stay in run_dir. OPAL 0.3.8 resolves the link
    (backends/claude.py: (Path(run_dir) / "agent-tmp").resolve()). Linux only: elsewhere there is no sandbox."""
    if not hasattr(os, 'getuid'):
        return
    base = os.path.join(tempfile.gettempdir(), 'claude-{}'.format(os.getuid()))
    os.makedirs(base, mode=0o700, exist_ok=True)
    target = tempfile.mkdtemp(prefix='opal-', dir=base)  # mode 0700, unique name
    if len(target) + 1 + SOCKET_NAME_MAX > SOCKET_PATH_MAX:
        raise RuntimeError('opal_wrap: {} is too long for the sandbox sockets'.format(target))
    os.makedirs(run_dir, exist_ok=True)
    os.symlink(target, os.path.join(run_dir, 'agent-tmp'))


def opal_wrap(optimizer, trainable, train_loader, opt):
    """Return (wrapped optimizer, scheduler). The scheduler steps once per update and reproduces adjust_learning_rate:
    epoch e (from 1) runs at learning_rate * lr_decay_rate ** #{m in lr_decay_epochs: e > m}."""
    import opal

    # OPAL records the parameters' devices here and restores them there when it undoes a failing program
    if torch.cuda.is_available() and any(not p.is_cuda for p in trainable.parameters()):
        raise RuntimeError('opal_wrap: move the trainable modules to the GPU before wrapping the optimizer')
    steps_per_epoch = min(len(train_loader), opt.max_batches) if opt.max_batches else len(train_loader)
    total_steps = opt.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[m * steps_per_epoch for m in opt.lr_decay_epochs], gamma=opt.lr_decay_rate)
    goal = ('{} student distilled from a frozen {} teacher on CIFAR-100 with {} '
            '(loss = {} * CE + {} * KL at T={} + {} * the {} loss), {} epochs of {} updates at batch {}; '
            'judged by test top-1 accuracy after the last epoch.').format(
        opt.model_s, opt.model_t, opt.distill.upper(), opt.gamma, opt.alpha, opt.kd_T, opt.beta, opt.distill,
        opt.epochs, steps_per_epoch, opt.batch_size)
    run_dir = os.path.join(opt.save_folder, 'opal')
    options = dict(backend=opt.opal_backend, max_cost_usd=opt.opal_max_cost_usd, decisions=opt.opal_decisions,
                   blocking=opt.opal_blocking == 'true', run_dir=run_dir,
                   epochs=opt.epochs, train_size=len(train_loader.dataset), batch_size=opt.batch_size)
    if opt.opal_backend == 'claude':
        options['backend_options'] = dict(unavailable_timeout=opt.opal_unavailable_timeout)
        short_agent_tmp(run_dir)
    optimizer = opal.wrap(optimizer, trainable, scheduler=scheduler, total_steps=total_steps, goal=goal,
                          **{key: value for key, value in options.items() if value is not None})
    return optimizer, scheduler
