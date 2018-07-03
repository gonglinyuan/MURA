import getopt
import os
import sys
import time

import torch

import train_multiview
from data_augmentation import DataTransform


def main():
    run_test("../../trained_models/03975-09025-densenet169large-adam-augrot30nobgpad/m-20180525-211346")


def run_test(path_model):
    model_name = 'DENSENET169-LARGE'
    model_pretrained = True
    path_data = '../../MURA-v1.0/'
    path_root = '../../'
    batch_size = 20
    img_size = 293
    crop_size = 256
    target_mean = 0.0
    target_std = 1.0
    data_transform = DataTransform(no_bg=True, pad=True)
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

    print('NN architecture = ', model_name)
    print("using data transforms: " + str(data_transform))

    if os.path.exists(path_model + "-L.pth.tar"):
        print('Testing the model with best valid-loss')
        train_multiview.test(
            path_data=path_data,
            path_root=path_root,
            path_model=path_model + "-L",
            model_name=model_name,
            model_pretrained=model_pretrained,
            batch_size=batch_size,
            device=device,
            transform=data_transform_valid
        )

    if os.path.exists(path_model + "-A.pth.tar"):
        print('Testing the model with best valid-auroc')
        train_multiview.test(
            path_data=path_data,
            path_root=path_root,
            path_model=path_model + "-A",
            model_name=model_name,
            model_pretrained=model_pretrained,
            batch_size=batch_size,
            device=device,
            transform=data_transform_valid
        )


# --------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
