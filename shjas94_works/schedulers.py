import torch
import torch.nn as nn
import torch.optim as optim

_scheduler_entrypoint = {
    'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR,
    'StepLR': optim.lr_scheduler.StepLR,
    'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau
}


def scheduler_entrypoint(scheduler_name):
    return _scheduler_entrypoint[scheduler_name]


def is_scheduler(scheduler_name):
    return scheduler_name in _scheduler_entrypoint


def create_scheduler(scheduler_name, **kwargs):
    if is_scheduler(scheduler_name):
        create_fn = scheduler_entrypoint(scheduler_name)
        scheduler = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown scheduler (%s)' % scheduler_name)

    return scheduler
