import time

import torch
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import convnet_models
import optimizers

__all__ = ["train", "test"]

CPU = torch.device("cpu")
GPU = torch.device("cuda:0")

POS_WEIGHT_TRAIN = torch.tensor(8280.0 / 5177.0).to(GPU)
POS_WEIGHT_VALID = torch.tensor(661.0 / 538.0).to(GPU)


def train(*, path_data_train, path_data_valid, path_log, path_model, config_train, config_valid):
    model = convnet_models.load(
        config_valid["model_name"],
        input_size=config_valid["crop_size"],
        pretrained=True
    ).to(GPU)
    data_loader_train = DataLoader(
        ImageFolder(path_data_train, transform=config_train["transform"].get(
            img_size=config_valid["img_size"],
            crop_size=config_valid["crop_size"]
        )),
        batch_size=config_train["batch_size"],
        shuffle=True,
        num_workers=10,
        pin_memory=True
    )
    data_loader_valid = DataLoader(
        ImageFolder(path_data_valid, transform=config_valid["transform"].get(
            img_size=config_valid["img_size"],
            crop_size=config_valid["crop_size"]
        )),
        batch_size=config_valid["batch_size"],
        shuffle=False,
        num_workers=10,
        pin_memory=True
    )
    if config_train["differential_lr"] > 1:
        optimizer, scheduler = optimizers.load_differential_lr(
            config_train["optimizer_name"],
            config_valid["model_name"],
            model,
            lr=config_train["learning_rate"],
            factor=config_train["differential_lr"],
            weight_decay=config_train["weight_decay"],
            nesterov=config_train["is_nesterov"],
            beta1=config_train["beta1"],
            beta2=config_train["beta2"],
            scheduling=config_train["scheduling"]
        )
    else:
        optimizer, scheduler = optimizers.load(
            config_train["optimizer_name"],
            model.parameters(),
            lr=config_train["learning_rate"],
            weight_decay=config_train["weight_decay"],
            nesterov=config_train["is_nesterov"],
            beta1=config_train["beta1"],
            beta2=config_train["beta2"],
            scheduling=config_train["scheduling"]
        )
    writer = SummaryWriter(log_dir=path_log)
    loss_fn_train = torch.nn.BCEWithLogitsLoss(reduction="elementwise_mean", pos_weight=POS_WEIGHT_TRAIN)
    loss_fn_valid = torch.nn.BCEWithLogitsLoss(reduction="elementwise_mean", pos_weight=POS_WEIGHT_VALID)
    loss_min, acc_max = epoch_valid(model, data_loader_valid, loss_fn_valid)
    timestamp_now = time.strftime("%Y%m%d") + '-' + time.strftime("%H%M%S")
    print(f"[000][------][{timestamp_now}]  valid-loss={loss_min:6f}  valid-acc={acc_max:6f}")
    writer.add_scalar("valid-loss", loss_min, global_step=0)
    writer.add_scalar("valid-acc", acc_max, global_step=0)
    for epoch in range(config_train["epoch_num"]):
        train_loss = epoch_train(model, data_loader_train, optimizer, loss_fn_train)
        valid_loss, valid_acc = epoch_valid(model, data_loader_valid, loss_fn_valid)
        writer.add_scalar("train-loss", train_loss, global_step=epoch + 1)
        writer.add_scalar("valid-loss", valid_loss, global_step=epoch + 1)
        writer.add_scalar("valid-acc", valid_acc, global_step=epoch + 1)
        scheduler.step(valid_loss)
        timestamp_now = time.strftime("%Y%m%d") + '-' + time.strftime("%H%M%S")
        save_flag, save_flag_l, save_flag_a = '----', '-', '-'
        if valid_loss < loss_min:
            save_flag, save_flag_l = 'save', 'L'
            loss_min = valid_loss
            torch.save({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "loss": valid_loss,
                "acc": valid_acc,
                "optimizer": optimizer.state_dict()
            }, path_model + "-L.pt")
        if valid_acc > acc_max:
            save_flag, save_flag_a = 'save', 'A'
            acc_max = valid_acc
            torch.save({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "loss": valid_loss,
                "acc": valid_acc,
                "optimizer": optimizer.state_dict()
            }, path_model + "-A.pt")
        print(f"[{epoch + 1:03d}][{save_flag + save_flag_l + save_flag_a}][{timestamp_now}]  train-loss={train_loss:6f}  valid-loss={valid_loss:6f}  valid-acc={valid_acc:6f}")


def find_lr(*, path_data_train, path_log, config_train, config_valid):
    model = convnet_models.load(
        config_valid["model_name"],
        input_size=config_valid["crop_size"],
        pretrained=True
    ).to(GPU)
    data_loader_train = DataLoader(
        ImageFolder(path_data_train, transform=config_train["transform"].get(
            img_size=config_valid["img_size"],
            crop_size=config_valid["crop_size"]
        )),
        batch_size=config_train["batch_size"],
        shuffle=True,
        num_workers=10,
        pin_memory=True
    )
    if config_train["differential_lr"] > 1:
        optimizer, scheduler = optimizers.load_finder_differential_lr(
            config_train["optimizer_name"],
            config_valid["model_name"],
            model,
            lr_min=1e-06,
            lr_max=0.01,
            num_epochs=config_train["epoch_num"],
            factor=config_train["differential_lr"],
            nesterov=config_train["is_nesterov"],
            beta1=config_train["beta1"],
            beta2=config_train["beta2"]
        )
    else:
        optimizer, scheduler = optimizers.load_finder(
            config_train["optimizer_name"],
            model.parameters(),
            lr_min=1e-06,
            lr_max=0.01,
            num_epochs=config_train["epoch_num"],
            nesterov=config_train["is_nesterov"],
            beta1=config_train["beta1"],
            beta2=config_train["beta2"]
        )
    writer = SummaryWriter(log_dir=path_log)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="elementwise_mean", pos_weight=POS_WEIGHT_TRAIN)
    for epoch in range(config_train["epoch_num"]):
        scheduler.step()
        train_loss = epoch_train(model, data_loader_train, optimizer, loss_fn)
        writer.add_scalar("train-loss", train_loss, global_step=epoch + 1)
        timestamp_now = time.strftime("%Y%m%d") + '-' + time.strftime("%H%M%S")
        print(f"[{epoch + 1:03d}][{timestamp_now}]  train-loss={train_loss:6f}")


def test(*, path_data, path_model, config_valid):
    cudnn.benchmark = True
    model = convnet_models.load(
        config_valid["model_name"],
        input_size=config_valid["crop_size"],
        pretrained=True
    ).to(GPU)
    model.load_state_dict(torch.load(path_model + ".pt")["state_dict"])
    data_loader = DataLoader(
        ImageFolder(path_data, transform=config_valid["transform"].get(
            img_size=config_valid["img_size"],
            crop_size=config_valid["crop_size"]
        )),
        batch_size=config_valid["batch_size"],
        shuffle=False,
        num_workers=10,
        pin_memory=True
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="elementwise_mean", pos_weight=POS_WEIGHT_VALID)
    loss, acc = epoch_valid(model, data_loader, loss_fn)
    timestamp_now = time.strftime("%Y%m%d") + '-' + time.strftime("%H%M%S")
    print(f"[----test---][{timestamp_now}]  valid-loss={loss:6f}  valid-acc={acc:6f}")
    return loss, acc


def epoch_train(model, data_loader, optimizer, loss_fn):
    model.train()
    count, total_loss = 0, 0.0
    for (x, y) in data_loader:
        x = x.to(GPU)
        y = y.to(torch.float32).to(GPU)
        loss = loss_fn(model(x).view(-1), y)
        total_loss += loss.item() * y.size(0)
        count += y.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / count


def epoch_valid(model, data_loader, loss_fn):
    model.eval()
    correct, count, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for (x, y) in data_loader:
            bs, n_crops, c, h, w = x.size()
            x, y = x.to(GPU), y.to(GPU)
            y_hat = model(x.view(-1, c, h, w))
            y_hat = y_hat.view(bs, n_crops).mean(1)
            total_loss += loss_fn(y_hat, y.to(torch.float32)).item() * y.size(0)
            count += y.size(0)
            predicted = (y_hat >= 0.0).to(torch.long)
            correct += (predicted == y).sum().item()
    return total_loss / count, correct / count
