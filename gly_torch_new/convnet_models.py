import types

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

__all__ = ["load"]


def load(model_name, input_size, pretrained):
    if model_name in ["DenseNet121", "DenseNet161", "DenseNet169", "DenseNet201"]:
        if model_name == "DenseNet121":
            model = torchvision.models.densenet121(pretrained=pretrained)
        elif model_name == "DenseNet161":
            model = torchvision.models.densenet161(pretrained=pretrained)
        elif model_name == "DenseNet169":
            model = torchvision.models.densenet169(pretrained=pretrained)
        elif model_name == "DenseNet201":
            model = torchvision.models.densenet201(pretrained=pretrained)
        else:
            raise Exception()

        kernel_count = model.classifier.in_features
        model.classifier = nn.Linear(kernel_count, 1)

        if input_size != 224:
            assert input_size % 32 == 0

            def forward(self, x):
                features = self.features(x)
                out = F.relu(features, inplace=True)
                out = F.avg_pool2d(out, kernel_size=input_size // 32, stride=1).view(features.size(0), -1)
                out = self.classifier(out)
                return out

            model.forward = types.MethodType(forward, model)
    else:
        raise Exception(f"Model {model_name} not found")
