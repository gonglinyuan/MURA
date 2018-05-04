import pretrainedmodels
import torch.nn as nn
import torchvision

MODELS = {
    'DENSENET121': torchvision.models.densenet121,
    'DENSENET161': torchvision.models.densenet161,
    'DENSENET169': torchvision.models.densenet169,
    'DENSENET201': torchvision.models.densenet201,
    'VGG11-BN': torchvision.models.vgg11_bn,
    'VGG13-BN': torchvision.models.vgg13_bn,
    'VGG16-BN': torchvision.models.vgg16_bn,
    'VGG19-BN': torchvision.models.vgg19_bn,
    'RESNET34': torchvision.models.resnet34,
    'RESNET50': torchvision.models.resnet50,
    'RESNET101': torchvision.models.resnet101,
    'RESNET152': torchvision.models.resnet152,
    'NASNETALARGE': pretrainedmodels.models.nasnetalarge,
    'NASNETAMOBILE': pretrainedmodels.models.nasnetamobile
}


class ConvnetModel(nn.Module):
    def __init__(self, model_name, *, class_count, is_trained):
        super(ConvnetModel, self).__init__()
        # load model and weights
        if model_name.startswith('NASNET'):
            self.convnet = MODELS[model_name](pretrained='imagenet+background')
        else:
            self.convnet = MODELS[model_name](pretrained=is_trained)
        # get input size of the last layer
        if model_name.startswith('DENSENET'):
            kernel_count = self.convnet.classifier.in_features
        elif model_name.startswith('VGG'):
            kernel_count = self.convnet.classifier[0].in_features
        elif model_name.startswith('RESNET'):
            kernel_count = self.convnet.fc.in_features
        elif model_name.startswith('NASNET'):
            kernel_count = self.convnet.last_linear.in_features
        else:
            print('ERROR')
            kernel_count = None
        # add last layer
        if model_name.startswith('RESNET'):
            self.convnet.fc = nn.Linear(kernel_count, class_count)
        elif model_name.startswith('NASNET'):
            self.convnet.last_linear = nn.Linear(kernel_count, class_count)
        else:
            self.convnet.classifier = nn.Linear(kernel_count, class_count)

    def forward(self, x):
        x = self.convnet(x)
        return x
