import sys

import numpy as np
import pandas
import torch

import predict
from data_augmentation import DataTransform


class Config:
    def __init__(self, *, model_name, file_model, batch_size, img_size, crop_size, target_mean, target_std,
                 data_transform, positions):
        self.model_name = model_name
        self.path_model = file_model
        self.batch_size = batch_size
        self.data_transform_test = data_transform.get_test(
            img_size=img_size, crop_size=crop_size, target_mean=target_mean, target_std=target_std, positions=positions)


def main():
    configs = [
        Config(
            model_name="SENET154",
            file_model="senet154",
            batch_size=32,
            img_size=256,
            crop_size=224,
            target_mean=np.array([0.485, 0.456, 0.406]),
            target_std=np.array([0.229, 0.224, 0.225]),
            data_transform=DataTransform(no_bg=True, pad=True),
            positions=[0, 6, 7, 3, 4]
        ), Config(
            model_name="SENET154-LARGE",
            file_model="senet154large",
            batch_size=16,
            img_size=293,
            crop_size=256,
            target_mean=np.array([0.485, 0.456, 0.406]),
            target_std=np.array([0.229, 0.224, 0.225]),
            data_transform=DataTransform(no_bg=True, pad=True),
            positions=[5, 1, 2, 3, 9]
        ), Config(
            model_name="INCEPTIONV4",
            file_model="inceptionv4",
            batch_size=32,
            img_size=341,
            crop_size=299,
            target_mean=0.5,
            target_std=0.5,
            data_transform=DataTransform(no_bg=True, pad=True),
            positions=[5, 6, 2, 8, 4]
        ), Config(
            model_name="INCEPTIONV4-LARGE",
            file_model="inceptionv4large",
            batch_size=16,
            img_size=378,
            crop_size=331,
            target_mean=0.5,
            target_std=0.5,
            data_transform=DataTransform(no_bg=True, pad=True),
            positions=[0, 6, 2, 3, 9]
        ), Config(
            model_name="INCEPTIONRESNETV2",
            file_model="inceptionresnetv2",
            batch_size=32,
            img_size=341,
            crop_size=299,
            target_mean=0.5,
            target_std=0.5,
            data_transform=DataTransform(no_bg=True, pad=True),
            positions=[0, 1, 7, 8, 9]
        ), Config(
            model_name="DENSENET201-LARGE3",
            file_model="densenet201",
            batch_size=32,
            img_size=366,
            crop_size=320,
            target_mean=0.456,
            target_std=0.225,
            data_transform=DataTransform(no_bg=True, pad=True),
            positions=[5, 1, 7, 8, 4]
        ), Config(
            model_name="DENSENET161-LARGE3",
            file_model="densenet161",
            batch_size=32,
            img_size=366,
            crop_size=320,
            target_mean=0.456,
            target_std=0.225,
            data_transform=DataTransform(no_bg=True, pad=True),
            positions=[0, 6, 7, 8, 4]
        ), Config(
            model_name="DENSENET169-LARGE3",
            file_model="densenet169",
            batch_size=32,
            img_size=366,
            crop_size=320,
            target_mean=0.456,
            target_std=0.225,
            data_transform=DataTransform(no_bg=True, pad=True),
            positions=[5, 6, 2, 3, 4]
        ), Config(
            model_name="NASNETALARGE",
            file_model="nasnetalarge",
            batch_size=8,
            img_size=354,
            crop_size=331,
            target_mean=0.5,
            target_std=0.5,
            data_transform=DataTransform(no_bg=True, pad=True),
            positions=[5, 1, 2, 8, 9]
        ), Config(
            model_name="PNASNET",
            file_model="pnasnet",
            batch_size=8,
            img_size=354,
            crop_size=331,
            target_mean=0.5,
            target_std=0.5,
            data_transform=DataTransform(no_bg=True, pad=True),
            positions=[0, 1, 7, 3, 9]
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
    num_views = 5
    output_lst = predict.predict(
        path_csv=path_csv,
        path_model="src/models/" + config.path_model,
        model_name=config.model_name,
        batch_size=config.batch_size,
        device=torch.device("cuda:0"),
        transform=config.data_transform_test
    )
    keys = []
    result = np.zeros((len(output_lst), num_views))
    for i in range(len(output_lst)):
        keys.append(output_lst[i][0])
        result[i, :] = np.array(output_lst[i][1:])
    return keys, result
    # return output_lst

    # pandas.DataFrame(output_lst).to_csv(path_output, header=False, index=False)


# --------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
