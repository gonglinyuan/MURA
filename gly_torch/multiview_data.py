import pandas
import torch.utils.data
from PIL import Image
import numpy as np


def img_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class MultiviewData(torch.utils.data.Dataset):
    def __init__(self, csv, img_prefix, transform=None):
        table = pandas.read_csv(csv, header=None).get_values()
        labels = {}
        image_paths = {}
        for path, label in table:
            prefix = '/'.join(path.split('/')[:-1])
            if prefix in labels:
                if labels[prefix] != label:
                    print("ERROR")
                    exit(-1)
                image_paths[prefix].append(path)
            else:
                labels[prefix] = label
                image_paths[prefix] = [path]
        self.samples = []
        for prefix, label in labels.items():
            self.samples.append((prefix, label, image_paths[prefix]))
        # np.random.shuffle(self.samples)
        self.loader = img_loader
        self.transform = transform
        self.img_prefix = img_prefix

    def __getitem__(self, index):
        sample_path = self.samples[index]
        sample_lst = []
        for path in sample_path[2]:
            sample = self.loader(self.img_prefix + path)
            if self.transform is not None:
                sample = self.transform(sample)
            sample_lst.append(sample)
        return torch.stack(sample_lst), sample_path[1]

    def __len__(self):
        return len(self.samples)
