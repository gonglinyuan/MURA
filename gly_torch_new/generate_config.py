import yaml

from data_augmentation import DataTransform

if __name__ == "__main__":
    config_train = {
        "transform": DataTransform(
            no_bg=True,
            pad=True,
            aug_rotate=30,
            flip_h="random",
            crop_mode="random"
        ),
        "batch_size": 20,
        "optimizer_name": "sgd",
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "is_nesterov": False,
        "epoch_num": 60
    }

    config_valid = {
        "model_name": "DenseNet201",
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

    with open("configs/config006.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
