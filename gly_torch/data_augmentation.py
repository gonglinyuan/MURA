import torch
import torchvision.transforms as transforms
import numpy as np

DATA_MEAN = 0.20558404267255
DATA_STD = 0.17694948680626902473216631207703
DEFAULT_IMG_SIZE = 256
DEFAULT_CROP_SIZE = 224
DEFAULT_NORMALIZE = transforms.Normalize([DATA_MEAN, DATA_MEAN, DATA_MEAN], [DATA_STD, DATA_STD, DATA_STD])


def default_transform(img_size=DEFAULT_IMG_SIZE, crop_size=DEFAULT_CROP_SIZE, target_mean=0.0, target_std=1.0):
    MEAN, STD = DATA_MEAN * np.ones(3), DATA_STD * np.ones(3)
    miu, sigma = MEAN - target_mean * STD / target_std, STD / target_std
    normalize = transforms.Normalize(miu.tolist(), sigma.tolist())
    return transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def default_transform_revised(img_size=DEFAULT_IMG_SIZE, crop_size=DEFAULT_CROP_SIZE, target_mean=0.0, target_std=1.0):
    MEAN, STD = DATA_MEAN * np.ones(3), DATA_STD * np.ones(3)
    miu, sigma = MEAN - target_mean * STD / target_std, STD / target_std
    normalize = transforms.Normalize(miu.tolist(), sigma.tolist())
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def valid_transform(img_size=DEFAULT_IMG_SIZE, crop_size=DEFAULT_CROP_SIZE, target_mean=0.0, target_std=1.0):
    MEAN, STD = DATA_MEAN * np.ones(3), DATA_STD * np.ones(3)
    miu, sigma = MEAN - target_mean * STD / target_std, STD / target_std
    normalize = transforms.Normalize(miu.tolist(), sigma.tolist())
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.TenCrop(crop_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
    ])


def augment_transform_slight(img_size=DEFAULT_IMG_SIZE, crop_size=DEFAULT_CROP_SIZE, target_mean=0.0, target_std=1.0):
    MEAN, STD = DATA_MEAN * np.ones(3), DATA_STD * np.ones(3)
    miu, sigma = MEAN - target_mean * STD / target_std, STD / target_std
    normalize = transforms.Normalize(miu.tolist(), sigma.tolist())
    return transforms.Compose([
        transforms.RandomAffine(degrees=20, shear=10),
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def augment_transform_slight_revised(img_size=DEFAULT_IMG_SIZE, crop_size=DEFAULT_CROP_SIZE, target_mean=0.0,
                                     target_std=1.0):
    MEAN, STD = DATA_MEAN * np.ones(3), DATA_STD * np.ones(3)
    miu, sigma = MEAN - target_mean * STD / target_std, STD / target_std
    normalize = transforms.Normalize(miu.tolist(), sigma.tolist())
    return transforms.Compose([
        transforms.RandomAffine(degrees=20, shear=10),
        transforms.Resize(img_size),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def augment_transform_rotation(img_size=DEFAULT_IMG_SIZE, crop_size=DEFAULT_CROP_SIZE, target_mean=0.0, target_std=1.0):
    MEAN, STD = DATA_MEAN * np.ones(3), DATA_STD * np.ones(3)
    miu, sigma = MEAN - target_mean * STD / target_std, STD / target_std
    normalize = transforms.Normalize(miu.tolist(), sigma.tolist())
    return transforms.Compose([
        transforms.RandomRotation(degrees=180),
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def augment_transform_rotation_warp(img_size=DEFAULT_IMG_SIZE, crop_size=DEFAULT_CROP_SIZE, target_mean=0.0,
                                    target_std=1.0):
    MEAN, STD = DATA_MEAN * np.ones(3), DATA_STD * np.ones(3)
    miu, sigma = MEAN - target_mean * STD / target_std, STD / target_std
    normalize = transforms.Normalize(miu.tolist(), sigma.tolist())
    return transforms.Compose([
        transforms.RandomAffine(degrees=180, shear=10),
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def augment_transform_rotation_warp_3(img_size=DEFAULT_IMG_SIZE, crop_size=DEFAULT_CROP_SIZE, target_mean=0.0,
                                    target_std=1.0):
    MEAN, STD = DATA_MEAN * np.ones(3), DATA_STD * np.ones(3)
    miu, sigma = MEAN - target_mean * STD / target_std, STD / target_std
    normalize = transforms.Normalize(miu.tolist(), sigma.tolist())
    return transforms.Compose([
        transforms.RandomAffine(degrees=180, shear=10),
        transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def augment_transform_slight_no_shear(img_size=DEFAULT_IMG_SIZE, crop_size=DEFAULT_CROP_SIZE, target_mean=0.0,
                                      target_std=1.0):
    MEAN, STD = DATA_MEAN * np.ones(3), DATA_STD * np.ones(3)
    miu, sigma = MEAN - target_mean * STD / target_std, STD / target_std
    normalize = transforms.Normalize(miu.tolist(), sigma.tolist())
    return transforms.Compose([
        transforms.RandomAffine(degrees=20),
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
