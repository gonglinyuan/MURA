import types

import pretrainedmodels.utils
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import densenet_custom

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
    'INCEPTIONRESNETV2': pretrainedmodels.models.inceptionresnetv2,
    'INCEPTIONV4': pretrainedmodels.models.inceptionv4,
    'INCEPTIONV3': torchvision.models.inception_v3,
    'XCEPTION': pretrainedmodels.models.xception,
    'RESNEXT101_64x4d': pretrainedmodels.models.resnext101_64x4d,
    'RESNEXT101_32x4d': pretrainedmodels.models.resnext101_32x4d,
    'VGG19-BN-LARGE': torchvision.models.vgg19_bn,
    'VGG16-BN-LARGE': torchvision.models.vgg16_bn,
    'DENSENET121-LARGE': torchvision.models.densenet121,
    'DENSENET161-LARGE': torchvision.models.densenet161,
    'DENSENET169-LARGE': torchvision.models.densenet169,
    'DENSENET201-LARGE': torchvision.models.densenet201,
    'DENSENET121-LARGE2': torchvision.models.densenet121,
    'DENSENET161-LARGE2': torchvision.models.densenet161,
    'DENSENET169-LARGE2': torchvision.models.densenet169,
    'DENSENET201-LARGE2': torchvision.models.densenet201,
    'DENSENET121-LARGE3': torchvision.models.densenet121,
    'DENSENET161-LARGE3': torchvision.models.densenet161,
    'DENSENET169-LARGE3': torchvision.models.densenet169,
    'DENSENET201-LARGE3': torchvision.models.densenet201,
    'INCEPTIONV4-LARGE': pretrainedmodels.models.inceptionv4,
    'POLYNET': pretrainedmodels.models.polynet,
    'PNASNET': pretrainedmodels.models.pnasnet5large,
    'DENSENET201-MODIFIED': densenet_custom.densenet201_modified,
    'SENET154-LARGE': pretrainedmodels.models.senet154,
    'DUALPATHNET107_5k-LARGE': pretrainedmodels.models.dpn107
}


class ConvnetModel(nn.Module):
    def __init__(self, model_name, *, class_count, is_trained):
        super(ConvnetModel, self).__init__()
        # load model and weights
        if (model_name.startswith('NASNET') or model_name == 'INCEPTIONRESNETV2' or model_name.startswith('INCEPTIONV4')
                or model_name == 'PNASNET'):
            self.convnet = MODELS[model_name](pretrained='imagenet+background')
        elif (model_name.startswith('SE') or model_name.startswith('RESNEXT') or model_name == 'XCEPTION' or
              model_name == "POLYNET"):
            self.convnet = MODELS[model_name](pretrained='imagenet')
        elif model_name == 'INCEPTIONV3':
            self.convnet = MODELS[model_name](pretrained='imagenet', transform_input=False, aux_logits=False)
        elif model_name.startswith('DUAL'):
            if model_name.endswith('5k') or model_name.endswith('5k-LARGE'):
                self.convnet = MODELS[model_name](pretrained='imagenet+5k')
            else:
                self.convnet = MODELS[model_name](pretrained='imagenet')
            if model_name.endswith('LARGE'):
                self.convnet.test_time_pool = False
        else:
            self.convnet = MODELS[model_name](pretrained=is_trained)
        # get input size of the last layer
        if model_name.startswith('DENSENET'):
            kernel_count = self.convnet.classifier.in_features
        elif model_name.startswith('DUAL'):
            kernel_count = self.convnet.classifier.in_channels
        elif model_name.startswith('VGG'):
            if model_name.endswith('LARGE'):
                kernel_count = 512 * 8 * 8
            else:
                kernel_count = self.convnet.classifier[0].in_features
        elif model_name.startswith('RESNET') or model_name == 'INCEPTIONV3':
            kernel_count = self.convnet.fc.in_features
        elif (model_name.startswith('NASNET') or model_name.startswith('SE') or model_name == 'INCEPTIONRESNETV2'
              or model_name.startswith('INCEPTIONV4') or model_name.startswith('RESNEXT') or model_name == 'XCEPTION'
              or model_name == 'POLYNET' or model_name == 'PNASNET'):
            kernel_count = self.convnet.last_linear.in_features
        else:
            print('ERROR')
            kernel_count = None
        # add last layer
        if model_name.startswith('RESNET') or model_name == 'INCEPTIONV3':
            self.convnet.fc = nn.Linear(kernel_count, class_count)
        elif model_name.startswith('SE'):
            if model_name.endswith('LARGE'):
                self.convnet.avgpool = nn.AvgPool2d(8, stride=1)
                kernel_count = kernel_count * 4
            self.convnet.last_linear = nn.Linear(kernel_count, class_count)
        elif (model_name.startswith('NASNET') or model_name == 'INCEPTIONRESNETV2'
              or model_name == 'INCEPTIONV4' or model_name.startswith('RESNEXT') or model_name == 'XCEPTION'
              or model_name == 'POLYNET' or model_name == 'PNASNET'):
            self.convnet.last_linear = nn.Linear(kernel_count, class_count)
        elif model_name.startswith('DUAL'):
            self.convnet.classifier = nn.Conv2d(kernel_count, class_count, kernel_size=1, bias=True)
        elif model_name.startswith('DENSENET') and model_name.endswith('LARGE'):
            self.convnet.classifier = nn.Linear(kernel_count, class_count)

            def forward(self, x):
                features = self.features(x)
                out = F.relu(features, inplace=True)
                out = F.avg_pool2d(out, kernel_size=8, stride=1).view(features.size(0), -1)
                out = self.classifier(out)
                return out

            self.convnet.forward = types.MethodType(forward, self.convnet)
        elif model_name.startswith('DENSENET') and model_name.endswith('LARGE2'):
            self.convnet.classifier = nn.Linear(kernel_count, class_count)

            def forward(self, x):
                features = self.features(x)
                out = F.relu(features, inplace=True)
                out = F.avg_pool2d(out, kernel_size=9, stride=1).view(features.size(0), -1)
                out = self.classifier(out)
                return out

            self.convnet.forward = types.MethodType(forward, self.convnet)
        elif model_name.startswith('DENSENET') and model_name.endswith('LARGE3'):
            self.convnet.classifier = nn.Linear(kernel_count, class_count)

            def forward(self, x):
                features = self.features(x)
                out = F.relu(features, inplace=True)
                out = F.avg_pool2d(out, kernel_size=10, stride=1).view(features.size(0), -1)
                out = self.classifier(out)
                return out

            self.convnet.forward = types.MethodType(forward, self.convnet)
        elif model_name.startswith('INCEPTIONV4') and model_name.endswith('LARGE'):
            self.convnet.avg_pool = nn.AvgPool2d(9, count_include_pad=False)
            self.convnet.last_linear = nn.Linear(kernel_count, class_count)
        else:
            self.convnet.classifier = nn.Linear(kernel_count, class_count)

    def forward(self, x):
        x = self.convnet(x)
        return x
