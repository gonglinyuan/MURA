import torch
from torch.utils.data import DataLoader

from convnet_models import ConvnetModel
from test_data import TestData

CPU = torch.device("cpu")


def predict(*, path_csv, path_model, model_name, model_pretrained, batch_size, device, transform):
    model = ConvnetModel(model_name, class_count=1, is_trained=model_pretrained).to(device)
    model_checkpoint = torch.load(path_model + ".pth.tar", map_location="cpu")
    model.load_state_dict(model_checkpoint['state_dict'])
    data_loader_test = DataLoader(
        TestData(path_csv, transform=transform),
        batch_size=batch_size, shuffle=False, num_workers=10)
    model.eval()
    study_y = {}
    for (x, study) in data_loader_test:
        bs, n_crops, c, h, w = x.size()
        x = x.to(device)
        with torch.no_grad():
            y = torch.nn.Sigmoid()(model(x.view(-1, c, h, w)))
        y = y.to(CPU)
        y = y.view(bs, n_crops).mean(1).view(-1)
        for i in range(bs):
            if study[i] in study_y:
                study_y[study[i]][0] += 1
                study_y[study[i]][1] += y[i]
            else:
                study_y[study[i]] = [1, y[i]]
    lst = []
    # arithmetic average
    for key, value in study_y.values():
        lst.append((key, value[1] / value[0]))
    return sorted(lst)
