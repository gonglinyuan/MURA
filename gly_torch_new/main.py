import sys
import time

import yaml

import train


def main():
    timestamp = time.strftime("%Y%m%d") + '-' + time.strftime("%H%M%S")

    path_data_train = '../../MURA_trainval_keras'
    path_data_valid = '../../MURA_valid1_keras'
    path_log = '../../trained_models_new/' + timestamp + '/tb'
    path_model = '../../trained_models_new/' + timestamp + '/m-' + timestamp
    path_config = sys.argv[1]

    with open(path_config, "r") as f:
        config = yaml.load(f)

    print(config)
    train.train(
        path_data_train=path_data_train,
        path_data_valid=path_data_valid,
        path_log=path_log,
        path_model=path_model,
        config_train=config["train"],
        config_valid=config["valid"]
    )

    print(config)
    print(timestamp)
    train.test(
        path_data=path_data_valid,
        path_model=path_model + "-L",
        config_valid=config["valid"]
    )

    train.test(
        path_data=path_data_valid,
        path_model=path_model + "-A",
        config_valid=config["valid"]
    )


if __name__ == "__main__":
    main()
