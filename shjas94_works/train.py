import os
from importlib import import_module
import random
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
import wandb
from dataset import CustomDataLoader
from utils import label_accuracy_score, add_hist
from losses import create_criterion
from optimizers import create_optimizer
from schedulers import create_scheduler
from cutmix import cutmix


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def collate_fn(batch):
    return tuple(zip(*batch))


def train(saved_dir, args, val_every=1):

    wandb.init(project='jaesub', entity='pstage12',
               group=args.model, name=args.run_name)

    wandb.config.update(args)
    print(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = '../input/data'
    # anns_file_path = dataset_path + '/' + 'train.json'
    train_path = dataset_path + '/train.json'
    val_path = dataset_path + '/val.json'
    test_path = dataset_path + '/test.json'
    seed_everything(args.seed)

    # collate_fn needs for batch

    train_transform = A.Compose([
                                ToTensorV2()
                                ])

    val_transform = A.Compose([
        ToTensorV2()
    ])

    test_transform = A.Compose([
        ToTensorV2()
    ])

    # create own Dataset 1 (skip)
    # validation set을 직접 나누고 싶은 경우
    # random_split 사용하여 data set을 8:2 로 분할
    # train_size = int(0.8*len(dataset))
    # val_size = int(len(dataset)-train_size)
    # dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=transform)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # create own Dataset 2
    # train dataset
    train_dataset = CustomDataLoader(
        data_dir=train_path, mode='train', transform=train_transform)

    # validation dataset
    val_dataset = CustomDataLoader(
        data_dir=val_path, mode='val', transform=val_transform)

    # DataLoader

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=4,
                                               collate_fn=collate_fn,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.val_batch_size,
                                             shuffle=False,
                                             num_workers=4,
                                             collate_fn=collate_fn,
                                             drop_last=True)

    model_module = getattr(import_module("model"), args.model)
    model = model_module(n_classes=12, n_blocks=[
                         3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]).to(device)
    wandb.watch(model)

    criterion = create_criterion(args.criterion)

    optimizer = create_optimizer(args.optimizer, params=model.parameters(
    ), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(
        args.scheduler, optimizer=optimizer, T_max=10)

    best_loss = 9999999
    best_mIoU = 0
    print('================Training Phase Started================')

    for epoch in range(args.num_epochs):
        train_loss_list = []
        model.train()

        for step, (images, masks, _) in enumerate(train_loader):
            # (batch, channel, height, width)
            images = torch.stack(images)
            # (batch, channel, height, width)
            masks = torch.stack(masks).long()

            if args.cutmix:
                images, masks = cutmix(images, masks, 1.0)

            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)

            # inference
            outputs = model(images)

            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.item())
            train_loss = np.mean(train_loss_list)
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch+1, args.num_epochs, step+1, len(train_loader), train_loss))
            wandb.log({
                "Train Loss": train_loss
            })
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            avrg_loss, mIoU = validation(
                epoch + 1, model, val_loader, criterion, device)
            # if avrg_loss < best_loss:
            if mIoU > best_mIoU:
                print('Best performance at epoch: {}'.format(epoch + 1))
                print('Save model in', saved_dir)
                # best_loss = avrg_loss
                best_mIoU = mIoU
                save_model(
                    model, saved_dir, f"{args.run_name}_{epoch+1}epoch_mIoU_{mIoU}.pt")
        scheduler.step()


def validation(epoch, model, data_loader, criterion, device):
    print('Start validation #{}'.format(epoch))
    model.eval()
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        mIoU_list = []
        hist = np.zeros((12, 12))
        for step, (images, masks, _) in enumerate(data_loader):

            # (batch, channel, height, width)
            images = torch.stack(images).to(device)
            # (batch, channel, height, width)
            masks = torch.stack(masks).long().to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1

            outputs = torch.argmax(
                outputs.squeeze(), dim=1).detach().cpu().numpy()

            # mIoU = label_accuracy_score(
            #     masks.detach().cpu().numpy(), outputs, n_class=12)[2]
            # mIoU_list.append(mIoU)

            hist = add_hist(hist, masks.detach().cpu().numpy(),
                            outputs, n_class=12)

        acc, acc_cls, mIoU, fwavacc = label_accuracy_score(hist)

        avrg_loss = total_loss / cnt

        print('Validation #{}  Average Loss: {:.4f}, mIoU: {:.4f}'.format(
            epoch, avrg_loss, mIoU))
        wandb.log({
            "Val mIoU": mIoU,
            "Val Loss": avrg_loss
        })
    return avrg_loss, mIoU


def save_model(model, saved_dir, file_name):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model.state_dict(), output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='number of train epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='input batch size for training (default: 8)')
    parser.add_argument('--val_batch_size', type=int, default=8,
                        help='input batch size for validation(default: 8)')
    parser.add_argument('--model', type=str, default='DeepLabV3',
                        help='model type & group name (default: DeepLabV3)')
    parser.add_argument('--criterion', type=str, default='cross_entropy',
                        help='criterion type (default: cross_entropy)')
    parser.add_argument('--optimizer', type=str, default='MADGRAD',
                        help='optimizer type (default: MADGRAD)')
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR',
                        help='scheduler type (default: CosineAnnealingLR')
    parser.add_argument('--lr', type=float, default=1e-04,
                        help='learning rate(default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-04)
    parser.add_argument('--cutmix', action='store_true')
    parser.add_argument('--run_name', type=str, required=True)
    args = parser.parse_args()

    val_every = 1

    saved_dir = './saved'
    if not os.path.isdir(saved_dir):
        os.mkdir(saved_dir)

    train(saved_dir, args, val_every)
