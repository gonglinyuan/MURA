import yaml

from data_augmentation import DataTransform

if __name__ == "__main__":
    config_train = {
        "img_size": 320,
        "crop_size": 320,
        "transform": DataTransform(
            no_bg=True,
            pad=True,
            flip_h="random",
            crop_mode="random"
        ),
        "batch_size": 16,
        "optimizer_name": "adam",
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "is_nesterov": False,
        "epoch_num": 60
    }

    config_valid = {
        "model_name": "DenseNet169",
        "img_size": 366,
        "crop_size": 320,
        "transform": DataTransform(
            no_bg=True,
            pad=True,
            crop_mode="ten"
        ),
        "batch_size": 16
    }

    config = {
        "train": config_train,
        "valid": config_valid
    }

    with open("configs/config001.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
