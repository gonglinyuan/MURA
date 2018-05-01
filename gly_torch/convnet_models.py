import torch.nn as nn
import torchvision


# class DenseNet121(nn.Module):
#     def __init__(self, class_count, is_trained):
#         super(DenseNet121, self).__init__()
#         self.densenet121 = torchvision.models.densenet121(pretrained=is_trained)
#         kernel_count = self.densenet121.classifier.in_features
#         self.densenet121.classifier = nn.Linear(kernel_count, class_count)
#
#     def forward(self, x):
#         x = self.densenet121(x)
#         return x
#
#
# class DenseNet169(nn.Module):
#     def __init__(self, class_count, is_trained):
#         super(DenseNet169, self).__init__()
#         self.densenet169 = torchvision.models.densenet169(pretrained=is_trained)
#         kernel_count = self.densenet169.classifier.in_features
#         self.densenet169.classifier = nn.Linear(kernel_count, class_count)
#
#     def forward(self, x):
#         x = self.densenet169(x)
#         return x
#
#
# class DenseNet201(nn.Module):
#     def __init__(self, class_count, is_trained):
#         super(DenseNet201, self).__init__()
#         self.densenet201 = torchvision.models.densenet201(pretrained=is_trained)
#         kernel_count = self.densenet201.classifier.in_features
#         self.densenet201.classifier = nn.Linear(kernel_count, class_count)
#
#     def forward(self, x):
#         x = self.densenet201(x)
#         return x
#
#
# class DenseNet161(nn.Module):
#     def __init__(self, class_count, is_trained):
#         super(DenseNet161, self).__init__()
#         self.densenet161 = torchvision.models.densenet161(pretrained=is_trained)
#         kernel_count = self.densenet161.classifier.in_features
#         self.densenet161.classifier = nn.Linear(kernel_count, class_count)
#
#     def forward(self, x):
#         x = self.densenet161(x)
#         return x


MODELS = {
    'DENSENET121': torchvision.models.densenet121,
    'DENSENET161': torchvision.models.densenet161,
    'DENSENET169': torchvision.models.densenet169,
    'DENSENET201': torchvision.models.densenet201
}


class ConvnetModel(nn.Module):
    def __init__(self, model_name, *, class_count, is_trained):
        super(ConvnetModel, self).__init__()
        self.convnet = MODELS[model_name](pretrained=is_trained)
        kernel_count = self.convnet.classifier.in_features
        self.convnet.classifier = nn.Linear(kernel_count, class_count)

    def forward(self, x):
        x = self.convnet(x)
        return x
