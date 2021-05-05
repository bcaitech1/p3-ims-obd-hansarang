import argparse
import glob
import os
import random
import re
from importlib import import_module
from pathlib import Path
import numpy as np
import segmentation_models_pytorch.losses
from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss
from utils import *
from loss import create_criterion
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import wandb
from loss import FocalLoss
import math
import torch.optim.lr_scheduler as lr_scheduler


class CosineAnnealingWarmUpRestart(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (
                    1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def collate_fn(batch):
    return tuple(zip(*batch))


def train(data_dir, model_dir, args):
    torch.backends.cudnn.benchmark = True
    train_path = data_dir + '/train.json'
    val_path = data_dir + '/val.json'
    seed_everything(args.seed)
    save_dir = './' + increment_path(os.path.join(model_dir, args.name))
    os.makedirs(save_dir)
    # -- settings
    use_cuda = torch.cuda.is_available()
    print("PyTorch version:[%s]." % (torch.__version__))
    device = torch.device('cuda' if use_cuda else 'cpu')
    print("device:[%s]." % (device))
    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: BaseAugmentation
    train_dataset = dataset_module(
        data_dir=train_path,
        mode='train'
    )
    val_dataset = dataset_module(
        data_dir=val_path,
        mode='val'
    )

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module()
    train_dataset.set_transform(transform)
    val_dataset.set_transform(transform)
    #train_dataset = CutMix(train_dataset, num_class=12, beta=1.0, prob=0.5, num_mix=2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.valid_batch_size,
        num_workers=8,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True
    )

    # -- model
    model_module = getattr(import_module("segmentation_models_pytorch"), args.model)  # default: BaseModel
    model = model_module(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        classes=12
    ).to(device)
    '''
    model_path = './model/zikgam2/best.pth'
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    wandb.watch(model)
    '''
    # checkpoint = torch.load(model_path, map_location=device)
    # model.load_state_dict(checkpoint)
    # model = torch.nn.DataParallel(model)

    # -- loss & metric

    # criterion = create_criterion(args.criterion).gamma = 0.5 # default: cross_entropy
    #criterion1 = FocalLoss(gamma=0.5)
    #criterion2 = segmentation_models_pytorch.losses.DiceLoss(mode='multiclass')
    criterion2 = segmentation_models_pytorch.losses.SoftCrossEntropyLoss(smooth_factor=0.1)
    #criterion3 = CutMixCrossEntropyLoss(True)
    opt_module = getattr(import_module("adamp"), args.optimizer)  # default: AdamP
    optimizer = opt_module(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        eps=1e-6
    )
    # optimizer = AdamP(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = CosineAnnealingWarmUpRestart(optimizer, T_0=4, T_mult=1, eta_max=2e-4,  T_up=1, gamma=0.75)
    #scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.65)
    # -- logging
    best_mIoU = 0
    best_val_loss = np.inf
    for epoch in range(1, args.epochs + 1):
        # train loop
        model.train()
        loss_value = 0
        for idx, (images, masks, _) in enumerate(tqdm(train_loader)):
            images = torch.stack(images)  # (batch, channel, height, width)
            masks = torch.stack(masks).long()
            images, masks = images.to(device), masks.to(device)


            outputs = model(images)
            optimizer.zero_grad()
            #loss1 = criterion1(outputs, masks)
            loss = criterion2(outputs, masks)
            #loss = 0.75*loss1+0.25*loss2
            loss.backward()
            optimizer.step()



            loss_value += loss.item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || lr {current_lr}"
                )
                wandb.log({"train loss": train_loss})
                loss_value = 0
        hist = np.zeros((12, 12))
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            for idx, (images, masks, _) in enumerate(val_loader):
                images = torch.stack(images)  # (batch, channel, height, width)
                masks = torch.stack(masks).long()
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                #loss1 = criterion1(outputs, masks)
                loss = criterion2(outputs, masks)
                #loss3 = criterion3(outputs, masks)
                #loss = 0.75*loss1+0.25*loss2
                val_loss_items.append(loss)
                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                hist = add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=12)
            val_loss = np.sum(val_loss_items) / len(val_loader)
            best_val_loss = min(best_val_loss, val_loss)
            _, _, mIoU, _ = label_accuracy_score(hist)
            wandb.log({
                "Test mIoU": mIoU,
                "Test Loss": val_loss})
            if mIoU > best_mIoU:
                print(f"New best model for val accuracy : {mIoU:4.2%}! saving the best model..")
                torch.save(model.state_dict(), f"{save_dir}/best.pth")
                best_mIoU = mIoU
            torch.save(model.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] mIoU : {mIoU:4.2%}, loss: {val_loss:4.2} || "
                f"best mIoU : {best_mIoU:4.2%}, best loss: {best_val_loss:4.2}"
            )
        scheduler.step()
        # val loop


if __name__ == '__main__':
    wandb.init(project='chowon', entity='pstage12')
    parser = argparse.ArgumentParser()
    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train (default: 28)')
    parser.add_argument('--dataset', type=str, default='TrashDataset',
                        help='dataset (default: TrashDataset)')
    parser.add_argument('--augmentation', type=str, default='NewAugmentation',
                        help='data augmentation type (default: NewAugmentation)')
    # parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 8)')
    parser.add_argument('--valid_batch_size', type=int, default=8,
                        help='input batch size for validing (default: 8)')
    parser.add_argument('--model', type=str, default='DeepLabV3Plus', help='model type (default: DeepLabV3Plus)')
    parser.add_argument('--encoder_name', type=str, default="timm-efficientnet-b5",
                        help='encoder type (default: timm-efficientnet-b5)')
    parser.add_argument('--encoder_weights', type=str, default="noisy-student",
                        help='encoder weight (default: noisy-student)')
    parser.add_argument('--optimizer', type=str, default='AdamP', help='optimizer type (default: AdamP)')
    parser.add_argument('--lr', type=float, default=5e-6, help='learning rate (default: 5e-6)')
    parser.add_argument('--lr_decay_step', type=int, default=1,
                        help='learning rate scheduler deacy step (default: 1)')
    # parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='focal',
                        help='criterion type (default: focal)')
    parser.add_argument('--log_interval', type=int, default=20,
                        help='how many batches to wait before logging training status')
    # parser.add_argument('--val_interval', type=int, default=150,
    # help='how many steps to calculate validataion')
    parser.add_argument('--name', default='zikgamwithsoftloss', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    args = parser.parse_args()

    wandb.run.name = 'zikgamwithoneloss'
    wandb.config.update(args)
    print(args)

    data_dir = os.environ.get('SM_CHANNEL_TRAIN', '../input/data')
    model_dir = os.environ.get('SM_MODEL_DIR', './model')

    train(data_dir, model_dir, args)