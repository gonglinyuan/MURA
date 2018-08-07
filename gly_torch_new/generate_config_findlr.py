import yaml
import numpy as np

from data_augmentation import DataTransform

if __name__ == "__main__":
    config_train = {
        "transform": DataTransform(
            no_bg=True,
            pad=True,
            aug_rotate=20,
            aug_shear=0,
            flip_h="random",
            crop_mode="random",
            random_crop_factor=0.08
        ),
        "batch_size": 16,
        "optimizer_name": "adam",
        "differential_lr": 3,
        "is_nesterov": False,
        "beta1": 0.9,
        "beta2": 0.999,
        "epoch_num": 80
    }

    config_valid = {
        "model_name": "SENet154",
        "img_size": 293,
        "crop_size": 256
    }

    config = {
        "train": config_train,
        "valid": config_valid
    }

    with open("configs/config056.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
