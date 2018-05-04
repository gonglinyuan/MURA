import torch
import torchvision.transforms as transforms

DATA_MEAN = 0.20558404267255
DATA_STD = 0.17694948680626902473216631207703
IMG_SIZE = 256
CROP_SIZE = 224
NORMALIZE = transforms.Normalize([DATA_MEAN, DATA_MEAN, DATA_MEAN], [DATA_STD, DATA_STD, DATA_STD])


def default_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([DATA_MEAN, DATA_MEAN, DATA_MEAN], [DATA_STD, DATA_STD, DATA_STD])
    ])

def default_transform_revised():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([DATA_MEAN, DATA_MEAN, DATA_MEAN], [DATA_STD, DATA_STD, DATA_STD])
    ])


def valid_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.TenCrop(CROP_SIZE),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([NORMALIZE(crop) for crop in crops]))
    ])


def augment_transform_slight():
    return transforms.Compose([
        transforms.RandomAffine(degrees=20, shear=10),
        transforms.RandomResizedCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        NORMALIZE
    ])

def augment_transform_slight_revised():
    return transforms.Compose([
        transforms.RandomAffine(degrees=20, shear=10),
        transforms.Resize(IMG_SIZE),
        transforms.RandomCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        NORMALIZE
    ])


def augment_transform_rotation():
    return transforms.Compose([
        transforms.RandomRotation(degrees=180),
        transforms.RandomResizedCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        NORMALIZE
    ])


def augment_transform_rotation_warp():
    return transforms.Compose([
        transforms.RandomAffine(degrees=180, shear=10),
        transforms.RandomResizedCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        NORMALIZE
    ])


def augment_transform_slight_no_shear():
    return transforms.Compose([
        transforms.RandomAffine(degrees=20),
        transforms.RandomResizedCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        NORMALIZE
    ])
