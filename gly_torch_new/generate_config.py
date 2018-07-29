import yaml
import numpy as np

from data_augmentation import DataTransform

if __name__ == "__main__":
    config_train = {
        "transform": DataTransform(
            no_bg=True,
            pad=True,
            aug_rotate=np.random.uniform(0.0, 30.0),
            flip_h="random",
            crop_mode="random",
            random_crop_factor=np.random.uniform(0.08, 0.5)
        ),
        "batch_size": 20,
        "optimizer_name": "adam",
        "learning_rate": 1e-4,
        "weight_decay": np.power(10.0, np.random.uniform(-6.0, -3.0)),
        "is_nesterov": False,
        "beta1": np.random.uniform(0.5, 0.9),
        "beta2": 0.999,
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

    with open("configs/config016.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
