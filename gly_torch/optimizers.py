from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


def adam_optimizers(parameters):
    optimizer = optim.Adam(parameters, lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min', verbose=True)
    return optimizer, scheduler


def sgd_optimizers(parameters):
    optimizer = optim.SGD(parameters, lr=0.001, momentum=0.9, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min', verbose=True)
    return optimizer, scheduler


def nsgd_optimizers(parameters):
    optimizer = optim.SGD(parameters, lr=0.001, momentum=0.9, weight_decay=1e-5, nesterov=True)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min', verbose=True)
    return optimizer, scheduler


def adam_optimizers_small(parameters):
    optimizer = optim.Adam(parameters, lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min', verbose=True)
    return optimizer, scheduler
