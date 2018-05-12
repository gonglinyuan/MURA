import pretrainedmodels
import torch.nn as nn
import torchvision
import pretrainedmodels.utils

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
    'NASNETAMOBILE': pretrainedmodels.models.nasnetamobile,
    'SENET154': pretrainedmodels.models.senet154,
    'SERESNEXT101_32x4d': pretrainedmodels.models.se_resnext101_32x4d,
    'SERESNEXT50_32x4d': pretrainedmodels.models.se_resnext50_32x4d,
    'DUALPATHNET107_5k': pretrainedmodels.models.dpn107,
    'DUALPATHNET131': pretrainedmodels.models.dpn131,
    'DUALPATHNET92_5k': pretrainedmodels.models.dpn92,
    'DUALPATHNET98': pretrainedmodels.models.dpn98,
    'INCEPTIONRESNETV2': pretrainedmodels.models.inceptionresnetv2
}


class ConvnetModel(nn.Module):
    def __init__(self, model_name, *, class_count, is_trained):
        super(ConvnetModel, self).__init__()
        # load model and weights
        if model_name.startswith('NASNET') or model_name.startswith('INCEPTION'):
            self.convnet = MODELS[model_name](pretrained='imagenet+background')
        elif model_name.startswith('SE'):
            self.convnet = MODELS[model_name](pretrained='imagenet')
        elif model_name.startswith('DUAL'):
            if model_name.endswith('5k'):
                self.convnet = MODELS[model_name](pretrained='imagenet+5k')
            else:
                self.convnet = MODELS[model_name](pretrained='imagenet')
        else:
            self.convnet = MODELS[model_name](pretrained=is_trained)
        # get input size of the last layer
        if model_name.startswith('DENSENET'):
            kernel_count = self.convnet.classifier.in_features
        elif model_name.startswith('DUAL'):
            kernel_count = self.convnet.classifier.in_channels
        elif model_name.startswith('VGG'):
            kernel_count = self.convnet.classifier[0].in_features
        elif model_name.startswith('RESNET'):
            kernel_count = self.convnet.fc.in_features
        elif model_name.startswith('NASNET') or model_name.startswith('SE') or model_name.startswith('INCEPTION'):
            kernel_count = self.convnet.last_linear.in_features
        else:
            print('ERROR')
            kernel_count = None
        # add last layer
        if model_name.startswith('RESNET'):
            self.convnet.fc = nn.Linear(kernel_count, class_count)
        elif model_name.startswith('NASNET') or model_name.startswith('SE') or model_name.startswith('INCEPTION'):
            self.convnet.last_linear = nn.Linear(kernel_count, class_count)
        elif model_name.startswith('DUAL'):
            self.convnet.classifier = nn.Conv2d(kernel_count, class_count, kernel_size=1, bias=True)
        else:
            self.convnet.classifier = nn.Linear(kernel_count, class_count)

    def forward(self, x):
        x = self.convnet(x)
        return x
