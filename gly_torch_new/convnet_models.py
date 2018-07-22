import types

import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["load"]


def load(model_name, input_size, pretrained):
    if model_name in ["DenseNet121", "DenseNet161", "DenseNet169", "DenseNet201"]:
        if model_name == "DenseNet121":
            model = pretrainedmodels.models.densenet121("imagenet" if pretrained else None)
        elif model_name == "DenseNet161":
            model = pretrainedmodels.models.densenet161("imagenet" if pretrained else None)
        elif model_name == "DenseNet169":
            model = pretrainedmodels.models.densenet169("imagenet" if pretrained else None)
        elif model_name == "DenseNet201":
            model = pretrainedmodels.models.densenet201("imagenet" if pretrained else None)
        else:
            raise Exception()

        kernel_count = model.last_linear.in_features
        model.last_linear = nn.Linear(kernel_count, 1)

        if input_size != model.input_size[1]:
            assert input_size % 32 == 0

            def logits(self, features):
                x = F.relu(features, inplace=True)
                x = F.avg_pool2d(x, kernel_size=input_size // 32, stride=1)
                x = x.view(x.size(0), -1)
                x = self.last_linear(x)
                return x

            model.logits = types.MethodType(logits, model)
    else:
        raise Exception("Model {} not found".format(model_name))
