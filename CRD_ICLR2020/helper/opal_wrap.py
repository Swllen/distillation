"""Wrap a student's optimizer with OPAL (https://github.com/xingjian-bai/opal): a coding agent rewrites the
optimizer while the model trains. Only used with train_student.py --opal."""
import os

import torch


def opal_wrap(optimizer, trainable, train_loader, opt):
    """Return (wrapped optimizer, scheduler). The scheduler steps once per update and reproduces adjust_learning_rate:
    epoch e (from 1) runs at learning_rate * lr_decay_rate ** #{m in lr_decay_epochs: e > m}."""
    import opal

    steps_per_epoch = min(len(train_loader), opt.max_batches) if opt.max_batches else len(train_loader)
    total_steps = opt.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[m * steps_per_epoch for m in opt.lr_decay_epochs], gamma=opt.lr_decay_rate)
    goal = ('{} student distilled from a frozen {} teacher on CIFAR-100 with {} '
            '(loss = {} * CE + {} * KL at T={} + {} * the {} loss), {} epochs of {} updates at batch {}; '
            'judged by test top-1 accuracy after the last epoch.').format(
        opt.model_s, opt.model_t, opt.distill.upper(), opt.gamma, opt.alpha, opt.kd_T, opt.beta, opt.distill,
        opt.epochs, steps_per_epoch, opt.batch_size)
    options = dict(backend=opt.opal_backend, max_cost_usd=opt.opal_max_cost_usd, decisions=opt.opal_decisions,
                   blocking=opt.opal_blocking == 'true', run_dir=os.path.join(opt.save_folder, 'opal'),
                   epochs=opt.epochs, train_size=len(train_loader.dataset), batch_size=opt.batch_size)
    if opt.opal_backend == 'claude':
        options['backend_options'] = dict(unavailable_timeout=opt.opal_unavailable_timeout)
    optimizer = opal.wrap(optimizer, trainable, scheduler=scheduler, total_steps=total_steps, goal=goal,
                          **{key: value for key, value in options.items() if value is not None})
    return optimizer, scheduler
