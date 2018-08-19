import numpy as np
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader

import convnet_models
from test_data import TestData

CPU = torch.device("cpu")
GPU = torch.device("cuda:0")


def predict(path_csv, config):
    cudnn.benchmark = True
    model = convnet_models.load(
        config["model_name"],
        input_size=config["crop_size"],
        pretrained=False
    ).to(GPU)
    model_checkpoint = torch.load(config["path_model"], map_location=lambda storage, loc: storage.cuda(0))
    model.load_state_dict(model_checkpoint['state_dict'])
    data_loader_test = DataLoader(
        TestData(path_csv, transform=config["transform"].get(
            img_size=config["img_size"],
            crop_size=config["crop_size"],
            target_mean=config["target_mean"],
            target_std=config["target_std"]
        )),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    model.eval()
    study_y = {}
    with torch.no_grad():
        for (x, study) in data_loader_test:
            bs, n_crops, c, h, w = x.size()
            x = x.to(GPU)
            y = model(x.view(-1, c, h, w))
            y = y.view(bs, n_crops).mean(1)
            for i in range(bs):
                if study[i] in study_y:
                    study_y[study[i]][0] += 1
                    study_y[study[i]][1] += y[i]
                else:
                    study_y[study[i]] = [1, y[i]]
    lst = []
    for key, value in study_y.items():
        lst.append([key, value[1] / value[0]])
    lst = sorted(lst)
    keys = []
    result = np.zeros(len(lst))
    for i in range(len(lst)):
        keys.append(lst[i][0])
        result[i] = lst[i][1]
    return keys, result
