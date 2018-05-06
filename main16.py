import getopt
import sys
import time
import numpy as np

import torch

import data_augmentation
import optimizers
import train


def main():
    # runTest()
    run_train()


def run_train():
    timestamp = time.strftime("%Y%m%d") + '-' + time.strftime("%H%M%S")
    path_data_train = '../../MURA_trainval_keras'
    path_data_valid = '../../MURA_valid1_keras'
    path_log = '../../trained_models/' + timestamp + '-seresnext50_32x4d-adam-augslight-revised/tb'
    model_name = 'SERESNEXT50_32x4d'
    model_pretrained = True
    batch_size = 16
    epoch_num = 100
    path_model = '../../trained_models/' + timestamp + '-seresnext50_32x4d-adam-augslight-revised/m-' + timestamp + '.pth.tar'

    device = None
    opts, _ = getopt.getopt(sys.argv[1:], "d:", ["device="])
    for opt, arg in opts:
        if opt in ("-d", "--device") and torch.cuda.is_available():
            device = torch.device("cuda:" + str(arg))
    if device is None:
        print("GPU not found! Using CPU!")
        device = torch.device("cpu")

    print('Training NN architecture = ', model_name)
    train.train(
        path_data_train=path_data_train,
        path_data_valid=path_data_valid,
        path_log=path_log,
        path_model=path_model,
        model_name=model_name,
        model_pretrained=model_pretrained,
        batch_size=batch_size,
        epoch_num=epoch_num,
        checkpoint=None,
        device=device,
            transform_train=data_augmentation.augment_transform_slight_revised(target_mean=np.array([0.485, 0.456, 0.406]), 
                                                                               target_std=np.array([0.229, 0.224, 0.225])),
        transform_valid=data_augmentation.valid_transform(target_mean=np.array([0.485, 0.456, 0.406]), 
                                                          target_std=np.array([0.229, 0.224, 0.225])),
        optimizer_fn=optimizers.adam_optimizers
    )

    print('Testing the trained model')
    train.test(
        path_data=path_data_valid,
        path_model=path_model,
        model_name=model_name,
        model_pretrained=model_pretrained,
        batch_size=batch_size,
        device=device,
        transform=data_augmentation.valid_transform()
    )


# def runTest():
#     pathDirData = './database'
#     pathFileTest = './dataset/test_1.txt'
#     nnArchitecture = 'DENSE-NET-121'
#     nnIsTrained = True
#     nnClassCount = 14
#     trBatchSize = 16
#     imgtransResize = 256
#     imgtransCrop = 224
#
#     pathModel = './models/m-25012018-123527.pth.tar'
#
#     timestampLaunch = ''
#
#     train.test(pathDirData, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize,
#                imgtransCrop, timestampLaunch)


# --------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
