import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import ImageFilter, ImageOps

DATA_MEAN = 0.20558404267255
DATA_STD = 0.17694948680626902473216631207703
DATA_MEAN_NO_BG = 0.2536153043662579
DATA_STD_NO_BG = 0.168205972008607
DEFAULT_IMG_SIZE = 256
DEFAULT_CROP_SIZE = 224
DEFAULT_NORMALIZE = transforms.Normalize([DATA_MEAN, DATA_MEAN, DATA_MEAN], [DATA_STD, DATA_STD, DATA_STD])


def get_normalize(target_mean, target_std):
    means, stds = DATA_MEAN * np.ones(3), DATA_STD * np.ones(3)
    miu, sigma = means - target_mean * stds / target_std, stds / target_std
    return transforms.Normalize(miu.tolist(), sigma.tolist())


def _remove_background(img):
    threshold = 8
    lookup = [256 - 1] * (256 - threshold) + [0] * threshold
    shape = img.size
    arr = np.array(img.convert('L')
                   .filter(ImageFilter.MedianFilter(size=3))
                   .filter(ImageFilter.MaxFilter(size=3))
                   .filter(ImageFilter.CONTOUR)
                   .crop([1, 1, shape[0] - 1, shape[1] - 1])
                   .point(lookup)
                   .filter(ImageFilter.MedianFilter(size=3))
                   .filter(ImageFilter.MaxFilter(size=3)))
    u, d, l, r = 0, arr.shape[0], 0, arr.shape[1]
    while u + 1 < d and np.max(arr[u, :]) == 0:
        u += 1
    while u + 1 < d and np.max(arr[d - 1, :]) == 0:
        d -= 1
    while l + 1 < r and np.max(arr[:, l]) == 0:
        l += 1
    while l + 1 < r and np.max(arr[:, r - 1]) == 0:
        r -= 1
    return img.crop([l, u, r, d])


def _pad(img):
    shape = img.size
    ratio = shape[0] / shape[1]
    if ratio < (2.0 / 3.0):
        w = int((shape[1] * (2.0 / 3.0) - shape[0]) / 2)
        img = ImageOps.expand(img, border=(w, 0, w, 0))
    elif ratio > (3.0 / 2.0):
        h = int((shape[0] * (2.0 / 3.0) - shape[1]) / 2)
        img = ImageOps.expand(img, border=(0, h, 0, h))
    return img


def get_normalize_no_bg(target_mean, target_std):
    means, stds = DATA_MEAN_NO_BG * np.ones(3), DATA_STD_NO_BG * np.ones(3)
    miu, sigma = means - target_mean * stds / target_std, stds / target_std
    return transforms.Normalize(miu.tolist(), sigma.tolist())


class DataTransform:
    def __init__(self, *, revised=False, aug=None, no_bg=False, pad=False, to_rgb=False):
        self.revised = revised
        if aug:
            self.aug = aug.lower()
        else:
            self.aug = None
        self.no_bg = no_bg
        self.pad = pad
        self.to_rgb = to_rgb

    def get_train(self, img_size=DEFAULT_IMG_SIZE, crop_size=DEFAULT_CROP_SIZE, target_mean=0.0, target_std=1.0):
        trans_list = []
        if self.to_rgb:
            trans_list.append(transforms.Lambda(lambda img: img.convert("RGB")))
        if self.no_bg:
            normalize = get_normalize_no_bg(target_mean, target_std)
            trans_list.append(transforms.Lambda(_remove_background))
            if self.pad:
                trans_list.append(transforms.Lambda(_pad))
        else:
            normalize = get_normalize(target_mean, target_std)
        if self.aug == "slight":
            trans_list.append(transforms.RandomAffine(degrees=20, shear=10))
        elif self.aug == "modified":
            trans_list.append(transforms.RandomAffine(degrees=15, shear=10))
            trans_list.append(transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05))
        elif self.aug == "rot30":
            trans_list.append(transforms.RandomRotation(degrees=30))
        if self.revised:
            trans_list.append(transforms.Resize(img_size))
            trans_list.append(transforms.RandomCrop(crop_size))
        else:
            trans_list.append(transforms.RandomResizedCrop(crop_size))
        trans_list.append(transforms.RandomHorizontalFlip())
        trans_list.append(transforms.ToTensor())
        trans_list.append(normalize)
        return transforms.Compose(trans_list)

    def get_valid(self, img_size=DEFAULT_IMG_SIZE, crop_size=DEFAULT_CROP_SIZE, target_mean=0.0, target_std=1.0):
        trans_list = []
        if self.to_rgb:
            trans_list.append(transforms.Lambda(lambda img: img.convert("RGB")))
        if self.no_bg:
            normalize = get_normalize_no_bg(target_mean, target_std)
            trans_list.append(transforms.Lambda(_remove_background))
            if self.pad:
                trans_list.append(transforms.Lambda(_pad))
        else:
            normalize = get_normalize(target_mean, target_std)
        trans_list.append(transforms.Resize(img_size))
        trans_list.append(transforms.TenCrop(crop_size))
        trans_list.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        trans_list.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        return transforms.Compose(trans_list)

    def __str__(self):
        s = ""
        if self.aug:
            s += "aug" + self.aug
        if self.revised:
            s += "revised"
        if self.no_bg:
            s += "nobg"
            if self.pad:
                s += "pad"
        if self.to_rgb:
            s += "2rgb"
        if s != "":
            s = "-" + s
        return s
