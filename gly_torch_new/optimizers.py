from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

__all__ = ["load", "load_differential_lr"]


def load(name, parameters, *, lr, weight_decay=1e-5, nesterov=False, beta1=0.9, beta2=0.999):
    if name == "adam":
        optimizer = optim.Adam(parameters, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    elif name == "sgd":
        optimizer = optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=nesterov)
    else:
        raise Exception("Optimizer {} not found".format(name))
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min', verbose=True)
    return optimizer, scheduler


def load_differential_lr(name, model_name, model, *, lr, lr_fc, weight_decay=1e-5, nesterov=False, beta1=0.9,
                         beta2=0.999):
    if model_name in ["VGG11bn", "VGG13bn", "VGG16bn", "VGG19bn"]:
        params_features = model.features.paramters()
        params_classifier = model.classifier.parameters()
    else:
        raise Exception()
    parameters = [
        {"params": params_features},
        {"params": params_classifier, "lr": lr_fc}
    ]
    if name == "adam":
        optimizer = optim.Adam(parameters, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    elif name == "sgd":
        optimizer = optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=nesterov)
    else:
        raise Exception("Optimizer {} not found".format(name))
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min', verbose=True)
    return optimizer, scheduler
