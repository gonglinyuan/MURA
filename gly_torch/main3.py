import getopt
import sys
import time

import torch

import data_augmentation
import optimizers
import train


def main():
    # runTest("../../trained_models/20180504-003410/m-20180504-003410.pth.tar")
    run_train()


def run_train():
    timestamp = time.strftime("%Y%m%d") + '-' + time.strftime("%H%M%S")
    model_name = 'DENSENET121'
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
        transform_train=data_augmentation.augment_transform_slight_no_bg(img_size=img_size, crop_size=crop_size,
                                                                         target_mean=target_mean,
                                                                         target_std=target_std),
        transform_valid=data_augmentation.valid_transform_no_bg(img_size=img_size, crop_size=crop_size,
                                                                target_mean=target_mean, target_std=target_std),
        optimizer_fn=optimizers.adam_optimizers
    )

    print('Testing the model with best valid-loss')
    print('timestamp = ' + timestamp)
    train.test(
        path_data=path_data_valid,
        path_model=path_model + "-L",
        model_name=model_name,
        model_pretrained=model_pretrained,
        batch_size=batch_size,
        device=device,
        transform=data_augmentation.valid_transform_no_bg(img_size=img_size, crop_size=crop_size,
                                                          target_mean=target_mean, target_std=target_std)
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
        transform=data_augmentation.valid_transform_no_bg(img_size=img_size, crop_size=crop_size,
                                                          target_mean=target_mean, target_std=target_std)
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
