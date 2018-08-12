import yaml

from data_augmentation import DataTransform

if __name__ == "__main__":
    config = {
        "model_name": "SENet154",
        "img_size": 293,
        "crop_size": 256,
        "transform": DataTransform(
            no_bg=True,
            pad=True,
            crop_mode="ten",
            ten_crop_positions=[0, 2, 9]
        ),
        "batch_size": 24,
        "path_model": "models/SENet154.pt"
    }

    with open("test_configs/config04.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
