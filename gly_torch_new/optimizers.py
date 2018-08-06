from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

__all__ = ["load", "load_differential_lr", "load_finder", "load_finder_differential_lr"]


def _differential_lr_parts(model_name, model):
    if model_name in ["VGG11bn", "VGG13bn", "VGG16bn", "VGG19bn"]:
        params_features = model.features.paramters()
        params_classifier = model.classifier.parameters()
    else:
        raise Exception()
    return params_features, params_classifier


def load(name, parameters, *, lr, weight_decay=1e-5, nesterov=False, beta1=0.9, beta2=0.999):
    if name == "adam":
        optimizer = optim.Adam(parameters, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    elif name == "sgd":
        optimizer = optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=nesterov)
    else:
        raise Exception("Optimizer {} not found".format(name))
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min', verbose=True)
    return optimizer, scheduler


def load_differential_lr(name, model_name, model, *, lr, factor, weight_decay=1e-5, nesterov=False, beta1=0.9,
                         beta2=0.999):
    params_features, params_classifier = _differential_lr_parts(model_name, model)
    parameters = [
        {"params": params_features},
        {"params": params_classifier, "lr": lr * factor}
    ]
    if name == "adam":
        optimizer = optim.Adam(parameters, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    elif name == "sgd":
        optimizer = optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=nesterov)
    else:
        raise Exception("Optimizer {} not found".format(name))
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min', verbose=True)
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
    params_features, params_classifier = _differential_lr_parts(model_name, model)
    parameters = [
        {"params": params_features},
        {"params": params_classifier, "lr": lr_min * factor}
    ]
    if name == "adam":
        optimizer = optim.Adam(parameters, lr=lr_min, betas=(beta1, beta2))
    elif name == "sgd":
        optimizer = optim.SGD(parameters, lr=lr_min, momentum=0.9, nesterov=nesterov)
    else:
        raise Exception("Optimizer {} not found".format(name))
    scheduler = ExponentialLR(optimizer, gamma=(lr_max / lr_min) ** (1 / num_epochs))
    return optimizer, scheduler
