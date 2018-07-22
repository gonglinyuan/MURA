import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import ImageFilter, ImageOps, Image

__all__ = ["DataTransform"]

DATA_MEAN = 0.20558404267255
DATA_STD = 0.17694948680626902473216631207703
DATA_MEAN_NO_BG = 0.2536153043662579
DATA_STD_NO_BG = 0.168205972008607


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


class HorizontalFlip(object):
    def __call__(self, img):
        return img, img.transpose(Image.FLIP_LEFT_RIGHT)


class DataTransform(object):
    def __init__(self, *, aug_rotate=0, aug_shear=0, no_bg=False, pad=False, flip_h=None, crop_mode=None,
                 random_crop_factor=0.08, ten_crop_positions=None, normalize=False):
        self.aug_rotate = aug_rotate
        self.aug_shear = aug_shear
        self.no_bg = no_bg
        self.pad = pad
        self.flip_h = flip_h
        self.crop_mode = crop_mode
        self.random_crop_factor = random_crop_factor
        self.ten_crop_positions = ten_crop_positions
        self.normalize = normalize
        self._num_crops = 0
        if self.crop_mode == "ten":
            self._num_crops = 10
        elif self.flip_h == "both":
            self._num_crops = 2

    def get(self, img_size, crop_size, target_mean=0.0, target_std=1.0):
        trans_list = []
        # Remove background and pad.
        if self.no_bg:
            trans_list.append(transforms.Lambda(_remove_background))
            if self.pad:
                trans_list.append(transforms.Lambda(_pad))
        # Apply random affine.
        if self.aug_rotate != 0 or self.aug_shear != 0:
            trans_list.append(transforms.RandomAffine(degrees=self.aug_rotate, shear=self.aug_shear))
        # Apply cropping.
        if self.crop_mode == "random":
            trans_list.append(transforms.RandomResizedCrop(crop_size, scale=(self.random_crop_factor, 1.0)))
        elif self.crop_mode == "ten":
            trans_list.append(transforms.Resize(img_size))
            trans_list.append(transforms.TenCrop(crop_size))
        else:
            trans_list.append(transforms.Resize((img_size, img_size)))
        # Apply horizontal flip.
        if self.flip_h == "random":
            trans_list.append(transforms.RandomHorizontalFlip())
        elif self.flip_h == "both":
            trans_list.append(HorizontalFlip())
        # Convert to torch tensor.
        if self._num_crops == 0:
            trans_list.append(transforms.ToTensor())
        else:
            trans_list.append(
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        # Select positions for TenCrop.
        if self.ten_crop_positions is not None:
            trans_list.append(transforms.Lambda(lambda crops: crops[self.ten_crop_positions]))
        # Normalize.
        if self.normalize:
            if self.no_bg:
                normalize = get_normalize_no_bg(target_mean, target_std)
            else:
                normalize = get_normalize(target_mean, target_std)
            if self._num_crops == 0:
                trans_list.append(normalize)
            else:
                trans_list.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        return transforms.Compose(trans_list)
