import sys

import pandas
import torch

import predict
from data_augmentation import DataTransform


def main():
    run_test("src/models/m-04051-09053", sys.argv[1], sys.argv[2])


def run_test(path_model, path_csv, path_output):
    model_name = 'VGG16-BN'
    model_pretrained = True
    batch_size = 16
    img_size = 256
    crop_size = 224
    target_mean = 0.0
    target_std = 1.0

    data_transform = DataTransform(no_bg=True, pad=True)
    data_transform_valid = data_transform.get_valid(img_size=img_size, crop_size=crop_size, target_mean=target_mean,
                                                    target_std=target_std)

    device = torch.device("cpu")

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
