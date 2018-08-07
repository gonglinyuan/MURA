import sys
import time

import yaml

import train


def main():
    timestamp = time.strftime("%Y%m%d") + '-' + time.strftime("%H%M%S")

    path_data_train = '../../MURA-v1.1-pytorch'
    path_log = '../../trained_models_new/' + timestamp + '/tb'
    path_config = sys.argv[1]

    with open(path_config, "r") as f:
        config = yaml.load(f)

    print(config)
    train.find_lr(
        path_data_train=path_data_train,
        path_log=path_log,
        config_train=config["train"],
        config_valid=config["valid"]
    )

    print(config)
    print(timestamp, path_config)


if __name__ == "__main__":
    main()
