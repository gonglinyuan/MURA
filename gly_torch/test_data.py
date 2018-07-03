import pandas
import torch.utils.data
from PIL import Image


def img_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class TestData(torch.utils.data.Dataset):
    def __init__(self, csv, transform=None):
        self.image_paths = list(pandas.read_csv(csv, header=None).get_values().reshape(-1))
        self.loader = img_loader
        self.transform = transform

    def __getitem__(self, index):
        print(2, index)
        path = self.image_paths[index]
        sample = self.loader(path)
        print(3, index)
        if self.transform is not None:
            sample = self.transform(sample)
        print(4, index)
        return sample, '/'.join(path.split('/')[:-1])

    def __len__(self):
        return len(self.image_paths)
