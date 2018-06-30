import time

import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics.ranking import roc_auc_score
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from convnet_models import ConvnetModel
from multiview_data import MultiviewData

CPU = torch.device("cpu")

EMA_ALPHA_TRAIN_LOSS = 0.99


def train(*, path_data, path_root, path_log, path_model, model_name, model_pretrained, batch_size,
          epoch_num, checkpoint, device, transform_train, transform_valid, optimizer_fn):
    model = ConvnetModel(model_name, class_count=1, is_trained=model_pretrained).to(device)
    data_loader_train = DataLoader(MultiviewData(path_data + "train.csv", path_root, transform=transform_train),
                                   batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    data_loader_valid = DataLoader(MultiviewData(path_data + "valid.csv", path_root, transform=transform_valid),
                                   batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    optimizer, scheduler = optimizer_fn(model.parameters())
    # ---- Load checkpoint
    if checkpoint:
        model_checkpoint = torch.load(checkpoint)
        model.load_state_dict(model_checkpoint['state_dict'])
        optimizer.load_state_dict(model_checkpoint['optimizer'])
    # Write Tensorboard
    writer = SummaryWriter(log_dir=path_log)
    # ---- TRAIN THE NETWORK
    loss_min, auroc_max = epoch_valid(model, data_loader_valid, device=device, max_batch_size=batch_size)
    print('Initial: valid-loss= ' + str(loss_min) + "  auroc= " + str(auroc_max))
    writer.add_scalar(tag="valid-loss", scalar_value=loss_min, global_step=0)
    writer.add_scalar(tag="valid-auroc", scalar_value=auroc_max, global_step=0)
    for epoch in range(0, epoch_num):
        train_loss = epoch_train(model, data_loader_train, optimizer, device=device, max_batch_size=batch_size)
        writer.add_scalar(tag="train-loss", scalar_value=train_loss, global_step=epoch + 1)
        loss, auroc = epoch_valid(model, data_loader_valid, device=device, max_batch_size=batch_size)
        timestamp_now = time.strftime("%Y%m%d") + '-' + time.strftime("%H%M%S")
        scheduler.step(loss)
        writer.add_scalar(tag="valid-loss", scalar_value=loss, global_step=epoch + 1)
        writer.add_scalar(tag="valid-auroc", scalar_value=auroc, global_step=epoch + 1)
        save_flag, save_flag_l, save_flag_a = '----', '-', '-'
        if loss < loss_min:
            save_flag, save_flag_l = 'save', 'L'
            loss_min = loss
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': loss_min,
                'optimizer': optimizer.state_dict()
            }, path_model + "-L.pth.tar")
        if auroc > auroc_max:
            save_flag, save_flag_a = 'save', 'A'
            auroc_max = auroc
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': loss_min,
                'optimizer': optimizer.state_dict()
            }, path_model + "-A.pth.tar")
        print('Epoch [{0}] [{1}] [{2}] train-loss={3}  valid-loss={4}  auroc={5}'.format(
            epoch + 1,
            save_flag + save_flag_l + save_flag_a,
            timestamp_now,
            train_loss,
            loss,
            auroc
        ))
    writer.close()


def epoch_train(model, data_loader, optimizer, device, max_batch_size):
    model.train()
    running_loss, running_norm = 0.0, 0.0
    num, idx = 0, 0
    rng = []
    labels = torch.zeros(max_batch_size)
    imgs = []
    for (xx, yy) in data_loader:
        xx = xx[0]
        yy = yy[0]
        print(xx.shape, yy.shape)
        sz = xx.shape[0]
        if num + sz > max_batch_size:
            x = torch.cat(imgs).to(device)
            imgs = []
            y_img = torch.nn.Sigmoid()(model(x).view(-1))
            y_hat = torch.empty(idx).to(device)
            for i in range(idx):
                y_hat[i] = torch.mean(y_img[rng[idx][0], rng[idx][1]])
            y = labels[:idx].to(torch.float32).to(device)
            loss_fn = torch.nn.BCELoss(size_average=True)
            loss = loss_fn(y_hat, y)
            running_loss = running_loss * EMA_ALPHA_TRAIN_LOSS + loss.item()
            running_norm = running_norm * EMA_ALPHA_TRAIN_LOSS + 1.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            rng[idx] = (num, num + sz)
            labels[idx] = yy
            imgs.append(xx)
            num += sz
            idx += 1
    return running_loss / running_norm


def epoch_valid(model, data_loader, device, max_batch_size):
    true = torch.LongTensor()
    score = torch.FloatTensor()
    bce = []
    model.eval()
    num, idx = 0, 0
    rng = []
    labels = torch.zeros(max_batch_size)
    imgs = []
    for (xx, yy) in data_loader:
        xx = xx[0]
        yy = yy[0]
        print(xx.shape, yy.shape)
        sz = xx.shape[0]
        if num + sz > max_batch_size:
            x = torch.cat(imgs)
            bs, n_crops, c, h, w = x.size()
            x = x.to(device)
            with torch.no_grad():
                y_img = torch.nn.Sigmoid()(model(x.view(-1, c, h, w)))
            y_img = y_img.to(CPU)
            y_img = y_img.view(bs, n_crops).mean(1)
            y_hat = torch.empty(idx).to(device)
            for i in range(idx):
                y_hat[i] = torch.mean(y_img[rng[idx][0], rng[idx][1]])
            y = labels[:idx].to(torch.float32).to(device)
            true = torch.cat((true, y), 0)
            score = torch.cat((score, y_hat), 0)
            loss_fn = torch.nn.BCELoss(size_average=True)
            bce.append(loss_fn(y_hat, y.to(torch.float32)))
        else:
            rng[idx] = (num, num + sz)
            labels[idx] = yy
            imgs.append(xx)
            num += sz
            idx += 1
    return torch.Tensor(bce).mean().item(), compute_auroc(true, score)


def compute_auroc(true, score):
    true = true.cpu().numpy()
    score = score.cpu().numpy()
    # roc_auc_score(true, score)
    return roc_auc_score(true, score)


def test(*, path_data, path_root, path_model, model_name, model_pretrained, batch_size, device, transform):
    cudnn.benchmark = True
    model = ConvnetModel(model_name, class_count=1, is_trained=model_pretrained).to(device)
    model_checkpoint = torch.load(path_model + ".pth.tar")
    model.load_state_dict(model_checkpoint['state_dict'])
    data_loader_test = DataLoader(MultiviewData(path_data + "valid.csv", path_root, transform=transform),
                                  batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    true = torch.LongTensor()
    score = torch.FloatTensor()
    bce = []
    model.eval()
    num, idx = 0, 0
    rng = []
    labels = torch.zeros(batch_size)
    imgs = []
    for (xx, yy) in data_loader_test:
        xx = xx[0]
        yy = yy[0]
        print(xx.shape, yy.shape)
        sz = xx.shape[0]
        if num + sz > batch_size:
            x = torch.cat(imgs)
            bs, n_crops, c, h, w = x.size()
            x = x.to(device)
            with torch.no_grad():
                y_img = torch.nn.Sigmoid()(model(x.view(-1, c, h, w)))
            y_img = y_img.to(CPU)
            y_img = y_img.view(bs, n_crops).mean(1)
            y_hat = torch.empty(idx).to(device)
            for i in range(idx):
                y_hat[i] = torch.mean(y_img[rng[idx][0], rng[idx][1]])
            y = labels[:idx].to(torch.float32).to(device)
            true = torch.cat((true, y), 0)
            score = torch.cat((score, y_hat), 0)
            loss_fn = torch.nn.BCELoss(size_average=True)
            bce.append(loss_fn(y_hat, y.to(torch.float32)))
        else:
            rng[idx] = (num, num + sz)
            labels[idx] = yy
            imgs.append(xx)
            num += sz
            idx += 1
    loss, auroc = torch.Tensor(bce).mean().item(), compute_auroc(true, score)
    auroc = compute_auroc(true, score)
    print('loss = ' + str(loss) + '  AUROC = ' + str(auroc))
    return
