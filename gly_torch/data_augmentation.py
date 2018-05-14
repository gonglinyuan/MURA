import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import ImageFilter

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


def default_transform(img_size=DEFAULT_IMG_SIZE, crop_size=DEFAULT_CROP_SIZE, target_mean=0.0, target_std=1.0):
    normalize = get_normalize(target_mean, target_std)
    return transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def default_transform_revised(img_size=DEFAULT_IMG_SIZE, crop_size=DEFAULT_CROP_SIZE, target_mean=0.0, target_std=1.0):
    normalize = get_normalize(target_mean, target_std)
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def valid_transform(img_size=DEFAULT_IMG_SIZE, crop_size=DEFAULT_CROP_SIZE, target_mean=0.0, target_std=1.0):
    normalize = get_normalize(target_mean, target_std)
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.TenCrop(crop_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
    ])


def augment_transform_slight(img_size=DEFAULT_IMG_SIZE, crop_size=DEFAULT_CROP_SIZE, target_mean=0.0, target_std=1.0):
    normalize = get_normalize(target_mean, target_std)
    return transforms.Compose([
        transforms.RandomAffine(degrees=20, shear=10),
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def augment_transform_slight_revised(img_size=DEFAULT_IMG_SIZE, crop_size=DEFAULT_CROP_SIZE, target_mean=0.0,
                                     target_std=1.0):
    normalize = get_normalize(target_mean, target_std)
    return transforms.Compose([
        transforms.RandomAffine(degrees=20, shear=10),
        transforms.Resize(img_size),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def augment_transform_rotation(img_size=DEFAULT_IMG_SIZE, crop_size=DEFAULT_CROP_SIZE, target_mean=0.0, target_std=1.0):
    normalize = get_normalize(target_mean, target_std)
    return transforms.Compose([
        transforms.RandomRotation(degrees=180),
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def augment_transform_rotation_warp(img_size=DEFAULT_IMG_SIZE, crop_size=DEFAULT_CROP_SIZE, target_mean=0.0,
                                    target_std=1.0):
    normalize = get_normalize(target_mean, target_std)
    return transforms.Compose([
        transforms.RandomAffine(degrees=180, shear=10),
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def augment_transform_rotation_warp_3(img_size=DEFAULT_IMG_SIZE, crop_size=DEFAULT_CROP_SIZE, target_mean=0.0,
                                      target_std=1.0):
    normalize = get_normalize(target_mean, target_std)
    return transforms.Compose([
        transforms.RandomAffine(degrees=180, shear=10),
        transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def augment_transform_slight_no_shear(img_size=DEFAULT_IMG_SIZE, crop_size=DEFAULT_CROP_SIZE, target_mean=0.0,
                                      target_std=1.0):
    normalize = get_normalize(target_mean, target_std)
    return transforms.Compose([
        transforms.RandomAffine(degrees=20),
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def remove_background(img):
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
    for i in range(arr.shape[0]):
        if np.max(arr[i, :]) == 0:
            u = i + 1
        else:
            break
    for i in reversed(range(arr.shape[0])):
        if np.max(arr[i, :]) == 0:
            d = i
        else:
            break
    for j in range(arr.shape[1]):
        if np.max(arr[:, j]) == 0:
            l = j + 1
        else:
            break
    for j in reversed(range(arr.shape[1])):
        if np.max(arr[:, j]) == 0:
            r = j
        else:
            break
    return img.crop([l, u, r, d])


def get_normalize_no_bg(target_mean, target_std):
    means, stds = DATA_MEAN_NO_BG * np.ones(3), DATA_STD_NO_BG * np.ones(3)
    miu, sigma = means - target_mean * stds / target_std, stds / target_std
    return transforms.Normalize(miu.tolist(), sigma.tolist())


def default_transform_no_bg(img_size=DEFAULT_IMG_SIZE, crop_size=DEFAULT_CROP_SIZE, target_mean=0.0, target_std=1.0):
    normalize = get_normalize_no_bg(target_mean, target_std)
    return transforms.Compose([
        transforms.Lambda(remove_background),
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def augment_transform_slight_no_bg(img_size=DEFAULT_IMG_SIZE, crop_size=DEFAULT_CROP_SIZE, target_mean=0.0,
                                   target_std=1.0):
    normalize = get_normalize_no_bg(target_mean, target_std)
    return transforms.Compose([
        transforms.Lambda(remove_background),
        transforms.RandomAffine(degrees=20, shear=10),
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def valid_transform_no_bg(img_size=DEFAULT_IMG_SIZE, crop_size=DEFAULT_CROP_SIZE, target_mean=0.0, target_std=1.0):
    normalize = get_normalize_no_bg(target_mean, target_std)
    return transforms.Compose([
        transforms.Lambda(remove_background),
        transforms.Resize(img_size),
        transforms.TenCrop(crop_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
    ])
