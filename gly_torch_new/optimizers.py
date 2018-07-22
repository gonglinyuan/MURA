from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

__all__ = ["load"]


def load(name, parameters, *, lr, weight_decay=1e-5, nesterov=False):
    if name == "adam":
        optimizer = optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        optimizer = optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=nesterov)
    else:
        raise Exception("Optimizer {} not found".format(name))
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min', verbose=True)
    return optimizer, scheduler
