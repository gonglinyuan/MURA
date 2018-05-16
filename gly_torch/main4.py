import getopt
import sys
import time

import torch

import optimizers
import train
from data_augmentation import DataTransform


def main():
    # runTest("../../trained_models/20180504-003410/m-20180504-003410.pth.tar")
    run_train()


def run_train():
    timestamp = time.strftime("%Y%m%d") + '-' + time.strftime("%H%M%S")
    model_name = 'VGG16-BN'
    model_pretrained = True
    path_data_train = '../../MURA_trainval_keras'
    path_data_valid = '../../MURA_valid1_keras'
    path_log = '../../trained_models/' + timestamp + '/tb'
    # batch_size = 16
    batch_size = 16
    epoch_num = 100
    img_size = 256
    crop_size = 224
    target_mean = 0.0
    target_std = 1.0
    path_model = '../../trained_models/' + timestamp + '/m-' + timestamp

    data_transform = DataTransform(revised=True, aug="slight", no_bg=True, pad=False, to_rgb=False)
    data_transform_train = data_transform.get_train(img_size=img_size, crop_size=crop_size, target_mean=target_mean,
                                                    target_std=target_std)
    data_transform_valid = data_transform.get_valid(img_size=img_size, crop_size=crop_size, target_mean=target_mean,
                                                    target_std=target_std)

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
        transform_train=data_transform_train,
        transform_valid=data_transform_valid,
        optimizer_fn=optimizers.adam_optimizers
    )

    print("using data transforms: " + str(data_transform))

    print('Testing the model with best valid-loss')
    print('timestamp = ' + timestamp)
    train.test(
        path_data=path_data_valid,
        path_model=path_model + "-L",
        model_name=model_name,
        model_pretrained=model_pretrained,
        batch_size=batch_size,
        device=device,
        transform=data_transform_valid
    )

    print('Testing the model with best valid-auroc')
    print('timestamp = ' + timestamp)
    train.test(
        path_data=path_data_valid,
        path_model=path_model + "-A",
        model_name=model_name,
        model_pretrained=model_pretrained,
        batch_size=batch_size,
        device=device,
        transform=data_transform_valid
    )


# def runTest(path_model):
#     path_data_valid = '../../MURA_valid1_keras'
#     model_name = 'INCEPTIONRESNETV2'
#     model_pretrained = True
#     # batch_size = 16
#     batch_size = 6
#     # batch_size = 10
#     img_size = 341
#     crop_size = 299
#     target_mean = 0.5
#     target_std = 0.5
#
#     device = None
#     opts, _ = getopt.getopt(sys.argv[1:], "d:", ["device="])
#     for opt, arg in opts:
#         if opt in ("-d", "--device") and torch.cuda.is_available():
#             device = torch.device("cuda:" + str(arg))
#     if device is None:
#         print("GPU not found! Using CPU!")
#         device = torch.device("cpu")
#
#     print('Testing the trained model')
#     train.test(
#         path_data=path_data_valid,
#         path_model=path_model,
#         model_name=model_name,
#         model_pretrained=model_pretrained,
#         batch_size=batch_size,
#         device=device,
#         transform=data_augmentation.valid_transform(img_size=img_size, crop_size=crop_size,
#                                                     target_mean=target_mean, target_std=target_std)
#     )


# --------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
