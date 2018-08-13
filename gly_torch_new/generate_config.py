import yaml
import numpy as np

from data_augmentation import DataTransform

if __name__ == "__main__":
    config_train = {
        "transform": DataTransform(
            no_bg=True,
            pad=True,
            aug_rotate=20,
            aug_shear=10,
            flip_h="random",
            crop_mode="random",
            random_crop_factor=0.08
        ),
        "batch_size": 20,
        "optimizer_name": "sgd",
        "learning_rate": 1e-6 * (0.01 / 1e-6) ** (71 / 80),
        "differential_lr": 1,
        "weight_decay": 0,
        "is_nesterov": False,
        "beta1": 0.9,
        "beta2": 0.999,
        "epoch_num": 60,
        "scheduling": "Cosine"
    }

    config_valid = {
        "model_name": "InceptionV4",
        "img_size": 378,
        "crop_size": 331,
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

    with open("configs/config095.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
