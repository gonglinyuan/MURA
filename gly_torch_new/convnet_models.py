import types

import pretrainedmodels.models
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
    elif model_name in ["VGG11bn", "VGG13bn", "VGG16bn", "VGG19bn"]:
        if model_name == "VGG11bn":
            model = torchvision.models.vgg11_bn(pretrained=pretrained)
        elif model_name == "VGG13bn":
            model = torchvision.models.vgg13_bn(pretrained=pretrained)
        elif model_name == "VGG16bn":
            model = torchvision.models.vgg16_bn(pretrained=pretrained)
        elif model_name == "VGG19bn":
            model = torchvision.models.vgg19_bn(pretrained=pretrained)
        else:
            raise Exception()

        model.classifier[-1] = nn.Linear(4096, 1)
    elif model_name in ["SENet154"]:
        if model_name == "SENet154":
            model = pretrainedmodels.models.senet154(pretrained="imagenet" if pretrained else None)
        else:
            raise Exception()

        kernel_count = model.last_linear.in_features
        model.last_linear = nn.Linear(kernel_count, 1)

        if input_size != 224:
            assert input_size % 32 == 0
            model.avg_pool = nn.AvgPool2d(input_size // 32, stride=1)
    elif model_name in ["DPN107"]:
        if model_name == "DPN107":
            model = pretrainedmodels.models.dpn107(pretrained="imagenet+5k" if pretrained else None)
        else:
            raise Exception()

        kernel_count = model.classifier.in_channels
        model.classifier = nn.Conv2d(kernel_count, 1, kernel_size=1, bias=True)

        assert input_size == 224
    elif model_name in ["InceptionV4"]:
        if model_name == "InceptionV4":
            model = pretrainedmodels.models.inceptionv4(pretrained="imagenet+background" if pretrained else None)
        else:
            raise Exception()

        kernel_count = model.last_linear.in_features
        model.last_linear = nn.Linear(kernel_count, 1)

        if input_size != 299:
            assert input_size % 32 == 11
            model.avg_pool = nn.AvgPool2d((input_size - 43) // 32, count_include_pad=False)
    elif model_name in ["NASNet", "PNASNet"]:
        if model_name == "NASNet":
            model = pretrainedmodels.models.nasnetalarge(pretrained="imagenet" if pretrained else None)
        elif model_name == "PNASNet":
            model = pretrainedmodels.models.pnasnet5large(pretrained="imagenet" if pretrained else None)
        else:
            raise Exception()

        kernel_count = model.last_linear.in_features
        model.last_linear = nn.Linear(kernel_count, 1)

        assert input_size == 331
    else:
        raise Exception(f"Model {model_name} not found")

    return model
