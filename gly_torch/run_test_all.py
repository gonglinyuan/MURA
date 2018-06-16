import pandas
import sys

import numpy as np
import torch

import predict
from data_augmentation import DataTransform


class Config:
    def __init__(self, *, model_name, file_model, batch_size, img_size, crop_size, target_mean, target_std, positions,
                 data_transform):
        self.model_name = model_name
        self.path_model = "src/models/" + file_model
        self.batch_size = batch_size
        self.data_transform_test = data_transform.get_test(
            img_size=img_size, crop_size=crop_size, target_mean=target_mean, target_std=target_std, positions=positions)


def main():
    configs = [
        Config(
            model_name="DUALPATHNET107_5k",
            file_model="m-03916-09049",
            batch_size=32,
            img_size=256,
            crop_size=224,
            target_mean=np.array([124 / 255, 117 / 255, 104 / 255]),
            target_std=1 / (.0167 * 255),
            positions=[0, 1, 8],
            data_transform=DataTransform(no_bg=True, pad=True)
        ), Config(
            model_name="SENET154",
            file_model="m-03924-09074",
            batch_size=32,
            img_size=256,
            crop_size=224,
            target_mean=np.array([0.485, 0.456, 0.406]),
            target_std=np.array([0.229, 0.224, 0.225]),
            positions=[0, 2, 9],
            data_transform=DataTransform(no_bg=True, pad=True)
        ), Config(
            model_name="INCEPTIONV4-LARGE",
            file_model="m-03879-09080",
            batch_size=32,
            img_size=378,
            crop_size=331,
            target_mean=0.5,
            target_std=0.5,
            positions=[1, 2, 9],
            data_transform=DataTransform(no_bg=True, pad=True)
        ), Config(
            model_name="VGG16-BN",
            file_model="m-04051-09053",
            batch_size=48,
            img_size=256,
            crop_size=224,
            target_mean=0.0,
            target_std=1.0,
            positions=[5, 6, 4],
            data_transform=DataTransform(no_bg=True, pad=True)
        ), Config(
            model_name="DENSENET201-LARGE3",
            file_model="m-04001-09065",
            batch_size=48,
            img_size=366,
            crop_size=320,
            target_mean=0.456,
            target_std=0.225,
            positions=[5, 7, 3],
            data_transform=DataTransform(no_bg=True, pad=True)
        ), Config(
            model_name="DENSENET161-LARGE3",
            file_model="m-03988-09084",
            batch_size=32,
            img_size=366,
            crop_size=320,
            target_mean=0.456,
            target_std=0.225,
            positions=[6, 3, 4],
            data_transform=DataTransform(no_bg=True, pad=True)
        ), Config(
            model_name="NASNETALARGE",
            file_model="m-03815-09099",
            batch_size=8,
            img_size=354,
            crop_size=331,
            target_mean=0.5,
            target_std=0.5,
            positions=[7, 8, 4],
            data_transform=DataTransform(no_bg=True, pad=True)
        )
    ]
    keys, results = [], []
    for config in configs:
        print(config.model_name)
        keys, result = run_test(sys.argv[1], config)
        results.append(result)
    results = np.concatenate(results, axis=1)
    score = np.mean(results, axis=1)
    label = np.array(score >= 0.5, dtype=np.int32)
    pandas.DataFrame(label, index=keys).to_csv(sys.argv[2], header=False)


def run_test(path_csv, config):
    output_lst = predict.predict(
        path_csv=path_csv,
        path_model=config.path_model,
        model_name=config.model_name,
        batch_size=config.batch_size,
        device=torch.device("cuda:0"),
        transform=config.data_transform_test
    )
    keys = []
    result = np.zeros((len(output_lst), 3))
    for i in range(len(output_lst)):
        keys.append(output_lst[i][0])
        result[i, :] = np.array(output_lst[i][1:])
    return keys, result
    # return output_lst

    # pandas.DataFrame(output_lst).to_csv(path_output, header=False, index=False)


# --------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
