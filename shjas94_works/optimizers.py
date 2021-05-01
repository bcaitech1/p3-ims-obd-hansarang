import torch
import torch.optim as optim
from madgrad import MADGRAD
from adamp import AdamP


_optimizer_entrypoints = {
    # 'cross_entropy': nn.CrossEntropyLoss,
    'Adam': optim.Adam,
    'MADGRAD': MADGRAD,
    'AdamP': AdamP

}


def optimizer_entrypoint(optimizer_name):
    return _optimizer_entrypoints[optimizer_name]


def is_optimizer(optimizer_name):
    return optimizer_name in _optimizer_entrypoints


def create_optimizer(optimizer_name, **kwargs):
    if is_optimizer(optimizer_name):
        create_fn = optimizer_entrypoint(optimizer_name)
        optimizer = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown optimizer (%s)' % optimizer_name)
    return optimizer
