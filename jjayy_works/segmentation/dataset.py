# 전처리를 위한 라이브러리
import os
import random
import time
import json
import warnings 
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms
import cv2

import numpy as np
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2



class BaseAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            #A.transforms.RandomContrast(limit=0.2, alway_apply=False, p=0.5),
            A.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
            A.random_flip(),
            A.RandomRotate90(),
            #Resize((256, 256), Image.BILINEAR),
            #Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            
            ToTensorV2(),
        ])

    def __call__(self, image, mask):
        return self.transform(image, mask)


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self):
        self.transform = transforms.Compose([
            CenterCrop((384, 320)),
            #Resize((256, 256), Image.BILINEAR),
            ColorJitter(0.1, 0.4, 0.2, 0.2),
            #RandAugment(),
            RandomHorizontalFlip(p=0.3),
            RandomRotation(10),
            ToTensorV2(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            #AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image)

# Randomize hue, vibrance, etc. 
 #'augment_photometric_distort': True, 
 # Have a chance to scale down the image and pad (to emulate smaller detections) 
 #'augment_expand': True, 
 # Potentialy sample a random crop from the image and put it in a random place 
 #'augment_random_sample_crop': True, 
 # Mirror the image with a probability of 1/2 
 #'augment_random_mirror': True, 
 # Flip the image vertically with a probability of 1/2 
 #'augment_random_flip': False, 
 # With uniform probability, rotate the image [0,90,180,270] degrees 
 #'augment_random_rot90': False, 


def eda(dataset_path):
    # %matplotlib inline
    
    anns_file_path = dataset_path + '/' + 'train_all.json'

    # Read annotations
    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())

    categories = dataset['categories']
    anns = dataset['annotations']
    imgs = dataset['images']
    nr_cats = len(categories)
    nr_annotations = len(anns)
    nr_images = len(imgs)

    # Load categories and super categories
    cat_names = []
    super_cat_names = []
    super_cat_ids = {}
    super_cat_last_name = ''
    nr_super_cats = 0
    for cat_it in categories:
        cat_names.append(cat_it['name'])
        super_cat_name = cat_it['supercategory']
        # Adding new supercat
        if super_cat_name != super_cat_last_name:
            super_cat_names.append(super_cat_name)
            super_cat_ids[super_cat_name] = nr_super_cats
            super_cat_last_name = super_cat_name
            nr_super_cats += 1
    
    # Count annotations
    cat_histogram = np.zeros(nr_cats,dtype=int)
    for ann in anns:
        cat_histogram[ann['category_id']] += 1

    # Convert to DataFrame
    df = pd.DataFrame({'Categories': cat_names, 'Number of annotations': cat_histogram})
    df = df.sort_values('Number of annotations', 0, False)
    
    # category labeling 
    sorted_temp_df = df.sort_index()

    # background = 0 에 해당되는 label 추가 후 기존들을 모두 label + 1 로 설정
    sorted_df = pd.DataFrame(["Backgroud"], columns = ["Categories"])
    sorted_df = sorted_df.append(sorted_temp_df, ignore_index=True)

    category_names = list(sorted_df.Categories)

    return category_names

def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

class CustomDataLoader(Dataset):
    """COCO format"""
    def __init__(self, data_dir, mode = 'train', transform = None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)

        
    def __getitem__(self, index: int):
        dataset_path = '../input/data'
        

        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0
        
        if (self.mode in ('train', 'val')):
            #category_names = eda(dataset_path)
            category_names = ['Backgroud','UNKNOWN','General trash','Paper','Paper pack','Metal','Glass','Plastic','Styrofoam','Plastic bag','Battery','Clothing']
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id + 1" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # Unknown = 1, General trash = 2, ... , Cigarette = 11
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = category_names.index(className)
                masks = np.maximum(self.coco.annToMask(anns[i])*pixel_value, masks)
            masks = masks.astype(np.float32)

            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            
            return images, masks, image_infos
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            
            return images, image_infos
    
    
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())
