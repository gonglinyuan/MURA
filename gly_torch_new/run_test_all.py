import sys

import numpy as np
import pandas
import torch

import yaml
import predict

from data_augmentation import DataTransform


def main():
    configs = [
        []
    ]
    keys, results = [], []
    for config in configs:
        print(config.model_name)
        keys, result = run_test(sys.argv[1], config)
        results.append(result)
    results = np.concatenate(results, axis=1)
    score = np.mean(results, axis=1)
    label = np.array(score >= 0.0, dtype=np.int32)
    pandas.DataFrame(label, index=keys).to_csv(sys.argv[2], header=False)


def run_test(path_csv, config):
    num_views = 5
    output_lst = predict.predict(
        path_csv=path_csv,
        path_model="models/" + config.path_model,
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
