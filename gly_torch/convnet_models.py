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
    'VGG19-BN': torchvision.models.vgg19_bn
}


class ConvnetModel(nn.Module):
    def __init__(self, model_name, *, class_count, is_trained):
        super(ConvnetModel, self).__init__()
        self.convnet = MODELS[model_name](pretrained=is_trained)
        if model_name.startswith('DENSENET'):
            kernel_count = self.convnet.classifier.in_features
        elif model_name.startswith('VGG'):
            kernel_count = self.convnet.classifier[0].in_features
        else:
            print('ERROR')
            kernel_count = None
        self.convnet.classifier = nn.Linear(kernel_count, class_count)

    def forward(self, x):
        x = self.convnet(x)
        return x
