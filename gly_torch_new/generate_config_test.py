import yaml

from data_augmentation import DataTransform

if __name__ == "__main__":
    config = {
        "model_name": "NASNet",
        "img_size": 354,
        "crop_size": 331,
        "transform": DataTransform(
            no_bg=True,
            pad=True,
            crop_mode="ten",
            ten_crop_positions=[7, 8, 4],
            normalize=True
        ),
        "batch_size": 8,
        "path_model": "models/NASNet.pt"
    }

    with open("test_configs/config07.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
