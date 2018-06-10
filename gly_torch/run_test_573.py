import sys

import pandas
import torch

import predict
from data_augmentation import DataTransform


def main():
    run_test("src/models/m-04001-09065", sys.argv[1], sys.argv[2])


def run_test(path_model, path_csv, path_output):
    model_name = 'DENSENET201-LARGE3'
    model_pretrained = True
    batch_size = 16
    img_size = 366
    crop_size = 320
    target_mean = 0.456
    target_std = 0.225

    data_transform = DataTransform(no_bg=True, pad=True)
    data_transform_valid = data_transform.get_test(img_size=img_size, crop_size=crop_size, target_mean=target_mean,
                                                   target_std=target_std, positions=[5, 7, 3])

    device = torch.device("cuda:0")

    output_lst = predict.predict(
        path_csv=path_csv,
        path_model=path_model,
        model_name=model_name,
        model_pretrained=model_pretrained,
        batch_size=batch_size,
        device=device,
        transform=data_transform_valid
    )

    pandas.DataFrame(output_lst).to_csv(path_output, header=False, index=False)


# --------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
