import yaml
import numpy as np

from data_augmentation import DataTransform

if __name__ == "__main__":
    config_train = {
        "transform": DataTransform(
            no_bg=True,
            pad=True,
            aug_rotate=np.random.uniform(0.0, 30.0),
            aug_shear=np.random.uniform(0.0, 15.0),
            flip_h="random",
            crop_mode="random",
            random_crop_factor=1.0 / np.random.uniform(2.0, 12.5)
        ),
        "batch_size": 20,
        "optimizer_name": "adam",
        "learning_rate": 3e-5,
        "weight_decay": np.power(10.0, np.random.uniform(-6.0, -3.0)).item(),
        "is_nesterov": False,
        "beta1": 1.0 - 1.0 / np.random.uniform(2.0, 10.0),
        "beta2": 0.999,
        "epoch_num": 60
    }

    config_valid = {
        "model_name": "VGG16bn",
        "img_size": 256,
        "crop_size": 224,
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

    with open("configs/config047.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
