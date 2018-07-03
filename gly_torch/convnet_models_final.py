import types

import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ConvnetModel(nn.Module):
    def __init__(self, model_name):
        super(ConvnetModel, self).__init__()
        if model_name == "SENET154":
            self.convnet = pretrainedmodels.models.senet154(pretrained=None)
            kernel_count = self.convnet.last_linear.in_features
            self.convnet.last_linear = nn.Linear(kernel_count, 1)
        elif model_name == "SENET154-LARGE":
            self.convnet = pretrainedmodels.models.senet154(pretrained=None)
            kernel_count = self.convnet.last_linear.in_features
            self.convnet.avgpool = nn.AvgPool2d(8, stride=1)
            self.convnet.last_linear = nn.Linear(kernel_count * 4, 1)
        elif model_name == "INCEPTIONV4":
            self.convnet = pretrainedmodels.models.inceptionv4(pretrained=None)
            kernel_count = self.convnet.last_linear.in_features
            self.convnet.last_linear = nn.Linear(kernel_count, 1)
        elif model_name == "INCEPTIONV4-LARGE":
            self.convnet = pretrainedmodels.models.inceptionv4(pretrained=None)
            kernel_count = self.convnet.last_linear.in_features
            self.convnet.avg_pool = nn.AvgPool2d(9, count_include_pad=False)
            self.convnet.last_linear = nn.Linear(kernel_count, 1)
        elif model_name == "INCEPTIONRESNETV2":
            self.convnet = pretrainedmodels.models.inceptionresnetv2(pretrained=False)
            kernel_count = self.convnet.last_linear.in_features
            self.convnet.last_linear = nn.Linear(kernel_count, 1)
        elif model_name == "DENSENET201-LARGE3":
            self.convnet = torchvision.models.densenet201(pretrained=False)
            kernel_count = self.convnet.classifier.in_features
            self.convnet.classifier = nn.Linear(kernel_count, 1)

            def forward(self, x):
                features = self.features(x)
                out = F.relu(features, inplace=True)
                out = F.avg_pool2d(out, kernel_size=10, stride=1).view(features.size(0), -1)
                out = self.classifier(out)
                return out

            self.convnet.forward = types.MethodType(forward, self.convnet)
        elif model_name == "DENSENET161-LARGE3":
            self.convnet = torchvision.models.densenet161(pretrained=False)
            kernel_count = self.convnet.classifier.in_features
            self.convnet.classifier = nn.Linear(kernel_count, 1)

            def forward(self, x):
                features = self.features(x)
                out = F.relu(features, inplace=True)
                out = F.avg_pool2d(out, kernel_size=10, stride=1).view(features.size(0), -1)
                out = self.classifier(out)
                return out

            self.convnet.forward = types.MethodType(forward, self.convnet)
        elif model_name == "DENSENET169-LARGE3":
            self.convnet = torchvision.models.densenet169(pretrained=False)
            kernel_count = self.convnet.classifier.in_features
            self.convnet.classifier = nn.Linear(kernel_count, 1)

            def forward(self, x):
                features = self.features(x)
                out = F.relu(features, inplace=True)
                out = F.avg_pool2d(out, kernel_size=10, stride=1).view(features.size(0), -1)
                out = self.classifier(out)
                return out

            self.convnet.forward = types.MethodType(forward, self.convnet)
        elif model_name == "NASNETALARGE":
            self.convnet = pretrainedmodels.models.nasnetalarge(pretrained=None)
            kernel_count = self.convnet.last_linear.in_features
            self.convnet.last_linear = nn.Linear(kernel_count, 1)
        elif model_name == "PNASNET":
            self.convnet = pretrainedmodels.models.pnasnet5large(pretrained=None)
            kernel_count = self.convnet.last_linear.in_features
            self.convnet.last_linear = nn.Linear(kernel_count, 1)

        else:
            raise Exception("model not found")

    def forward(self, x):
        x = self.convnet(x)
        return x
