import os
import random
import time
import json
import warnings 
warnings.filterwarnings('ignore')
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
import pandas as pd
from importlib import import_module
from pathlib import Path
from tqdm import tqdm

from adamp import AdamP
import segmentation_models_pytorch as smp 

import sklearn
from sklearn.model_selection import StratifiedKFold

import wandb

from model import SegNet, DeepLabV3Plus
from dataset import CustomDataLoader
from loss import create_criterion, custom_loss
from utils import label_accuracy_score


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

def save_model(model, saved_dir, file_name):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model.state_dict(), output_path)

def collate_fn(batch):
    return tuple(zip(*batch))


def train(data_dir, saved_dir, args):
    seed_everything(args.seed)

    print('pytorch version: {}'.format(torch.__version__))
    print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

    print(torch.cuda.get_device_name(0))
    print(torch.cuda.device_count())

    # -- gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    '''
    # -- model dict save directory
    if not os.path.isdir(saved_dir):                                                           
        os.mkdir(saved_dir)

    # -- dataset
    anns_file_path = data_dir + '/' + 'train_all.json'
    # Read annotations
    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())

    anno=[]
    for k in range(len(dataset['annotations'])):
        anno.append(dataset['annotations'][k]['category_id'])

    skf = StratifiedKFold(n_splits=args.split, shuffle=True, random_state=args.seed)
    folds=[]
    for fold_index, (trn_idx, val_idx) in enumerate(skf.split(dataset['annotations'], anno),1):
        folds.append((trn_idx, val_idx))

    
    # -- augmentation
    #transform_module = getattr(import_module("dataset"), args.augmentation[1])  # 0: CustomAugmentation
    #transform = transform_module()
    transform = A.Compose([
                        #A.Resize(128, 128),
                        ToTensorV2(),
                        ])

    # total dataset
    total_dataset = CustomDataLoader(data_dir=anns_file_path, mode='train', transform=transform)

    '''

    best_models = [] # 폴드별로 가장 validation mIoU가 높은 모델 저장
    for fold in range(args.split):
        print(f'[folds: {fold}]')
        # cuda cache 초기화
        torch.cuda.empty_cache()
        if fold is 0:
            continue
        if fold is 1:
            continue

        '''
        # get image number from splited annotation info(train:folds[fold][0])
        data_train_images = []
        for k in folds[fold][0]:
            a = dataset['annotations'][k]['image_id']
            for k in range(len(dataset['images'])):
                if dataset['images'][k]['id'] == a:
                    data_train_images.append(k)
        # get image number from splited annotation info(val:folds[fold][1])
        data_val_images = []
        for k in folds[fold][1]:
            a = dataset['annotations'][k]['image_id']
            for k in range(len(dataset['images'])):
                if dataset['images'][k]['id'] == a:
                    data_val_images.append(k)

        print(len(data_train_images))
        tmp = set(data_train_images)
        train_set = list(tmp)
        print(len(train_set))

        print(len(data_val_images))
        tmp2 = set(data_val_images)
        val_set = list(tmp2)
        print(len(val_set))

        
        train_dataset = torch.utils.data.Subset(total_dataset, train_set)
        val_dataset = torch.utils.data.Subset(total_dataset, val_set)
        '''
        
        trainset_path = data_dir + '/' + f'train_data{fold}.json'
        validset_path = data_dir + '/' + f'valid_data{fold}.json'

        base_transform = A.Compose([
                    A.Resize(256, 256),
                    A.transforms.Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
                    ToTensorV2(),
                    ])

        train_transform = A.Compose([
                    A.Resize(256, 256),
                    #A.RandomBrightnessContrast(p=0.3),
                    A.transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.2, hue=0.2, p=0.3),
                    #A.CLAHE(p=0.3),
                    A.HorizontalFlip(p=0.3),
                    A.Rotate(p=0.3, limit=45),
                    A.transforms.Cutout(num_holes=8, max_h_size=10, max_w_size=10, p=0.5),
                    #A.random_flip(),
                    #A.RandomRotate90(),
                    A.transforms.Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
                    ToTensorV2(),
                    ])

        #dataset
        train_dataset = CustomDataLoader(data_dir=trainset_path, mode='train', transform=train_transform)
        val_dataset = CustomDataLoader(data_dir=validset_path, mode='val', transform=base_transform)


        # -- data_loader
        train_loader = DataLoader(dataset=train_dataset, 
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    drop_last=False,
                                    num_workers=4,
                                    collate_fn=collate_fn)

        val_loader = DataLoader(dataset=val_dataset, 
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=4,
                                    collate_fn=collate_fn)
        print('data loader finished')

        # -- model
        aux_params=dict(
            pooling='avg',
            dropout=args.dropout,
            activation=None,
            classes=12,
        )

        model = DeepLabV3Plus(args.backbone, aux_params)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to(device)

        #wandb.watch(model)

        # -- loss & metric
        optimizer = AdamP(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
        #optimizer = torch.optim.Adam(params = model.parameters(), lr = args.lr, weight_decay=1e-4)
        #scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
        scheduler = CosineAnnealingLR(optimizer, T_max=1, eta_min=5e-6, last_epoch=-1)
        #criterion = smp.losses.FocalLoss('multiclass')
        criterion = smp.losses.SoftCrossEntropyLoss(smooth_factor=0.1)

        print('Start training..')
        best_loss = 9999999
        best_mIoU = 0
        for epoch in range(args.epoch):
            model.train()
            for step, (images, masks, _) in enumerate(tqdm(train_loader)):
                images = torch.stack(images)       # (batch, channel, height, width)
                masks = torch.stack(masks).long()  # (batch, channel, height, width)
                
                # gpu 연산을 위해 device 할당
                images, masks = images.to(device), masks.to(device)

                # inference
                outputs = model(images)
                
                (a, b)=outputs  #a는 위치 예측, b는 라벨 예측
                #loss = custom_loss(a, masks, 2) #0:Jaccard, 1:Dice, 2:Focal, 3:Lovasz
                loss = criterion(a, masks)

                # loss 계산 (cross entropy loss)
                #loss = criterion(outputs, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # step 주기에 따른 loss 출력
                if (step + 1) % args.log_interval == 0: #batch8-25
                    train_loss = loss.item()
                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch}/{args.epoch}]({step + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || lr {current_lr}"
                    )
                    wandb.log({
                    "Train Loss": train_loss,
                    })
            
            scheduler.step()

            # val loop
            with torch.no_grad():
                print('Start validation #{}'.format(epoch))
                model.eval()
                total_loss = 0
                cnt = 0
                mIoU_list = []
                for step, (images, masks, _) in enumerate(tqdm(val_loader)):
                    
                    images = torch.stack(images)       # (batch, channel, height, width)
                    masks = torch.stack(masks).long()  # (batch, channel, height, width)

                    images, masks = images.to(device), masks.to(device)            

                    outputs = model(images)
                    (a, b) = outputs
                    #loss = custom_loss(a, masks, 2) #0:Jaccard, 1:Dice, 2:Focal, 3:Lovasz
                    loss = criterion(a, masks)
                    total_loss += loss
                    cnt += 1
                    
                    a = torch.argmax(a.squeeze(), dim=1).detach().cpu().numpy()

                    mIoU = label_accuracy_score(masks.detach().cpu().numpy(), a, n_class=12)[2]
                    mIoU_list.append(mIoU)
                    wandb.log({
                        "Validation Loss": loss,
                        "Validation mIoU": mIoU
                    })
                    
                avrg_loss = total_loss / cnt
                mean_mIoU = np.mean(mIoU_list)


                # validation 주기에 따른 loss 출력 및 best model 저장

                if mean_mIoU > best_mIoU:
                    best_mIoU = mean_mIoU
                    print(f"New best mIoU at epoch{epoch} : {mean_mIoU:4.4}!")
                    print(f"val loss: {avrg_loss:4.4}")
                    print('Save model in', saved_dir)
                    if avrg_loss < best_loss:
                        best_loss = avrg_loss
                    save_model(model, saved_dir, f"n_{fold}_epoch{epoch}_{avrg_loss:4.4}_{mean_mIoU:4.4}.pt")

                print('Validation #{}  Average Loss: {:.4f}, mIoU: {:.4f}'.format(epoch, avrg_loss, mean_mIoU))
                print(
                    f"[Val] Average Loss: {avrg_loss:4.4}, mIoU {mean_mIoU:4.4} || "
                    f"Best Loss: {best_loss:4.4}, Best mIoU: {best_mIoU:4.4}"
                )
                wandb.log({
                    "Validation avg Loss": avrg_loss,
                    "Validation avg mIoU": mean_mIoU
                })
                print()



if __name__ == '__main__':
    wandb.init(project="jjay")
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--split', type=int, default=5, help='number of split for skfold')
    parser.add_argument('--epoch', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='train_all', help='dataset')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument('--batch_size', type=int, default=7, help='input batch size for training (default: 64)')
    parser.add_argument('--model', type=str, default='DeepLabV3Plus', help='model type (default: DeepLabV3Plus)')
    parser.add_argument('--backbone', type=str, default='timm-efficientnet-b0', help='pretrained backbone model')
    parser.add_argument('--dropout', type=float, default=0.7, help='dropout ratio')
    parser.add_argument('--optimizer', type=str, default='AdamP', help='optimizer type (default: Adam)') #not used
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR', help='optimizer type (default: StepLR)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--criterion', type=str, default='SoftCrossEntrophy', help='criterion type (default: cross_entropy)') #not used
    parser.add_argument('--lr_decay_step', type=int, default=2, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=12, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='DeepLabV3Plus-timm-efficientnet-b5', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data'))
    parser.add_argument('--saved_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './saved'))

    args = parser.parse_args()
    print(args)
    wandb.run.name=args.name
    wandb.run.save()
    wandb.config.update(args)



    data_dir = args.data_dir
    saved_dir = args.saved_dir

    train(data_dir, saved_dir, args)