import time

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics.ranking import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from convnet_models import DenseNet121
from convnet_models import DenseNet169
from convnet_models import DenseNet201

DATA_MEAN = 0.20558404267255
DATA_STD = 0.17694948680626902473216631207703

CPU = torch.device("cpu")
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def train(path_data_train, path_data_valid, model_name, model_pretrained, batch_size, epoch_num, img_size, crop_size,
          timestamp, checkpoint):
    # -------------------- SETTINGS: NETWORK ARCHITECTURE
    if model_name == 'DENSE-NET-121':
        model = DenseNet121(class_count=1, is_trained=model_pretrained).to(DEVICE)
    elif model_name == 'DENSE-NET-169':
        model = DenseNet169(class_count=1, is_trained=model_pretrained).to(DEVICE)
    elif model_name == 'DENSE-NET-201':
        model = DenseNet201(class_count=1, is_trained=model_pretrained).to(DEVICE)
    else:
        print('Error: architecture not found')
        return
    # -------------------- SETTINGS: DATA TRANSFORMS
    normalize = transforms.Normalize([DATA_MEAN, DATA_MEAN, DATA_MEAN], [DATA_STD, DATA_STD, DATA_STD])
    transform_sequence_train = transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    transform_sequence_valid = transforms.Compose([
        transforms.Resize(img_size),
        transforms.TenCrop(crop_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
    ])
    # -------------------- SETTINGS: DATASET BUILDERS
    data_loader_train = DataLoader(
        ImageFolder(path_data_train, transform=transform_sequence_train),
        batch_size=batch_size, shuffle=True, num_workers=20, pin_memory=True)
    data_loader_valid = DataLoader(
        ImageFolder(path_data_valid, transform=transform_sequence_valid),
        batch_size=batch_size, shuffle=False, num_workers=20, pin_memory=True)
    # -------------------- SETTINGS: OPTIMIZER & SCHEDULER
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min')
    # -------------------- SETTINGS: LOSS
    loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)
    # ---- Load checkpoint
    if checkpoint:
        model_checkpoint = torch.load(checkpoint)
        model.load_state_dict(model_checkpoint['state_dict'])
        optimizer.load_state_dict(model_checkpoint['optimizer'])
    # ---- TRAIN THE NETWORK
    loss_min, auroc_max = epoch_valid(model, data_loader_valid)
    print('Initial loss=' + str(loss_min))
    for epoch in range(0, epoch_num):
        epoch_train(model, data_loader_train, optimizer, loss_fn)
        loss, auroc = epoch_valid(model, data_loader_valid)
        timestamp_now = time.strftime("%Y%m%d") + '-' + time.strftime("%H%M%S")
        scheduler.step(loss.data[0])
        if loss.data[0] < loss_min:
            loss_min = loss.data[0]
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_loss': loss_min,
                        'optimizer': optimizer.state_dict()}, 'm-' + timestamp + '.pth.tar')
            print('Epoch [' + str(epoch + 1) + '] [save] [' + timestamp_now + '] loss= ' + str(
                loss.data[0]) + '  auroc = ' + str(auroc))
        else:
            print('Epoch [' + str(epoch + 1) + '] [----] [' + timestamp_now + '] loss= ' + str(
                loss.data[0]) + '  auroc = ' + str(auroc))


def epoch_train(model, data_loader, optimizer, loss_fn):
    model.train()
    for (x, y) in data_loader:
        x = x.to(DEVICE)
        y = y.to(torch.float32).to(DEVICE)
        loss = loss_fn(model(x).view(-1), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def epoch_valid(model, data_loader):
    # model.eval()
    # num, loss_value, loss_sum = 0, 0, 0
    # for (x, y) in data_loader:
    #     x = x.to(DEVICE)
    #     y = y.to(DEVICE)
    #     with torch.no_grad():
    #         loss = loss_fn(model(x).view(-1), y)
    #     loss_sum += loss
    #     loss_value += loss.data[0]
    #     num += 1
    # return loss_value / num, loss_sum / num
    true = torch.LongTensor()
    score = torch.FloatTensor()
    bce = []
    model.eval()
    for (x, y) in data_loader:
        bs, n_crops, c, h, w = x.size()
        x = x.to(DEVICE)
        true = torch.cat((true, y), 0)
        with torch.no_grad():
            y_hat = torch.nn.Sigmoid()(model(x.view(-1, c, h, w)))
        y_hat = y_hat.view(bs, n_crops, -1).mean(1)
        bce.append(torch.nn.BCELoss(size_average=True)(y_hat, y.to(torch.float32).to(DEVICE)).to(CPU))
        score = torch.cat((score, y_hat), 0)
    return torch.Tensor(bce).mean(), compute_auroc(true, score)


def compute_auroc(true, score):
    true = true.cpu().numpy()
    score = score.cpu().numpy()
    roc_auc_score(true, score)
    return roc_auc_score(true, score)


def test(path_data, path_model, model_name, model_pretrained, batch_size, img_size, crop_size, timestamp):
    cudnn.benchmark = True
    # -------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
    if model_name == 'DENSE-NET-121':
        model = DenseNet121(class_count=1, is_trained=model_pretrained).to(DEVICE)
    elif model_name == 'DENSE-NET-169':
        model = DenseNet169(class_count=1, is_trained=model_pretrained).to(DEVICE)
    elif model_name == 'DENSE-NET-201':
        model = DenseNet201(class_count=1, is_trained=model_pretrained).to(DEVICE)
    else:
        print('Error: architecture not found')
        return
    model_checkpoint = torch.load(path_model)
    model.load_state_dict(model_checkpoint['state_dict'])
    # -------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
    normalize = transforms.Normalize([DATA_MEAN, DATA_MEAN, DATA_MEAN], [DATA_STD, DATA_STD, DATA_STD])
    transform_sequence_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.TenCrop(crop_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
    ])
    # -------------------- SETTINGS: DATASET BUILDERS
    data_loader_test = DataLoader(
        ImageFolder(path_data, transform=transform_sequence_test),
        batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True)
    true = torch.FloatTensor().to(DEVICE)
    score = torch.FloatTensor().to(DEVICE)
    model.eval()
    for (x, y) in data_loader_test:
        bs, n_crops, c, h, w = x.size()
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        true = torch.cat((true, y), 0)
        with torch.no_grad():
            y_hat = torch.nn.Sigmoid()(model(x.view(-1, c, h, w)))
        y_hat = y_hat.view(bs, n_crops, -1).mean(1)
        score = torch.cat((score, y_hat.data), 0)
    auroc = compute_auroc(true, score)
    print('AUROC = ', auroc)
    return
