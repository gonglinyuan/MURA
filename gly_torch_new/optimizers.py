import math

from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import _LRScheduler

__all__ = ["load", "load_differential_lr", "load_finder", "load_finder_differential_lr"]


class CosineAnnealingLRRestart(_LRScheduler):
    def __init__(self, optimizer, period, eta_min=0, last_epoch=-1):
        self.period = period
        self.eta_min = eta_min
        super(CosineAnnealingLRRestart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * (self.last_epoch % self.period) / self.period)) / 2
                for base_lr in self.base_lrs]


def _differential_lr_params(model_name, model, lr, factor):
    if model_name in ["VGG11bn", "VGG13bn", "VGG16bn", "VGG19bn"]:
        return [
            {"params": model.features.parameters()},
            {"params": model.classifier.parameters(), "lr": lr * factor}
        ]
    elif model_name in ["SENet154"]:
        return [
            {"params": model.layer0.parameters(), "lr": lr / factor},
            {"params": model.layer1.parameters()},
            {"params": model.layer2.parameters()},
            {"params": model.layer3.parameters()},
            {"params": model.layer4.parameters()},
            {"params": model.last_linear.parameters(), "lr": lr * factor}
        ]
    elif model_name in ["DPN107"]:
        return [
            {"params": model.features.parameters()},
            {"params": model.classifier.parameters(), "lr": lr * factor}
        ]
    elif model_name in ["InceptionV4"]:
        return [
            {"params": model.features.parameters()},
            {"params": model.last_linear.parameters(), "lr": lr * factor}
        ]
    else:
        raise Exception()


def load(name, parameters, *, lr, weight_decay=1e-5, nesterov=False, beta1=0.9, beta2=0.999, scheduling="Adaptive"):
    if name == "adam":
        optimizer = optim.Adam(parameters, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    elif name == "sgd":
        optimizer = optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=nesterov)
    else:
        raise Exception("Optimizer {} not found".format(name))
    if scheduling == "Adaptive":
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min', verbose=True)
    elif scheduling == "Cosine":
        scheduler = CosineAnnealingLRRestart(optimizer, 30)
    else:
        raise Exception()
    return optimizer, scheduler


def load_differential_lr(name, model_name, model, *, lr, factor, weight_decay=1e-5, nesterov=False, beta1=0.9,
                         beta2=0.999, scheduling="Adaptive"):
    parameters = _differential_lr_params(model_name, model, lr, factor)
    if name == "adam":
        optimizer = optim.Adam(parameters, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    elif name == "sgd":
        optimizer = optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=nesterov)
    else:
        raise Exception("Optimizer {} not found".format(name))
    if scheduling == "Adaptive":
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min', verbose=True)
    elif scheduling == "Cosine":
        scheduler = CosineAnnealingLRRestart(optimizer, 30)
    else:
        raise Exception()
    return optimizer, scheduler


def load_finder(name, parameters, *, lr_min, lr_max, num_epochs, nesterov=False, beta1=0.9, beta2=0.999):
    if name == "adam":
        optimizer = optim.Adam(parameters, lr=lr_min, betas=(beta1, beta2))
    elif name == "sgd":
        optimizer = optim.SGD(parameters, lr=lr_min, momentum=0.9, nesterov=nesterov)
    else:
        raise Exception("Optimizer {} not found".format(name))
    scheduler = ExponentialLR(optimizer, gamma=(lr_max / lr_min) ** (1 / num_epochs))
    return optimizer, scheduler


def load_finder_differential_lr(name, model_name, model, *, lr_min, lr_max, num_epochs, factor, nesterov=False,
                                beta1=0.9, beta2=0.999):
    parameters = _differential_lr_params(model_name, model, lr_min, factor)
    if name == "adam":
        optimizer = optim.Adam(parameters, lr=lr_min, betas=(beta1, beta2))
    elif name == "sgd":
        optimizer = optim.SGD(parameters, lr=lr_min, momentum=0.9, nesterov=nesterov)
    else:
        raise Exception("Optimizer {} not found".format(name))
    scheduler = ExponentialLR(optimizer, gamma=(lr_max / lr_min) ** (1 / num_epochs))
    return optimizer, scheduler
