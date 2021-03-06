import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

dataset_path = '../input/data'
anns_file_path = dataset_path + '/' + 'train.json'

__all__ = ['CustomDataLoader']


def read_data(anns_file_path):
    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())

    categories = dataset['categories']
    anns = dataset['annotations']
    imgs = dataset['images']
    return categories, anns, imgs


def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return "None"


def get_category_names():
    categories, anns, imgs = read_data(anns_file_path)
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

    cat_histogram = np.zeros(nr_cats, dtype=int)
    for ann in anns:
        cat_histogram[ann['category_id']] += 1
    df = pd.DataFrame(
        {'Categories': cat_names, 'Number of annotations': cat_histogram})
    df = df.sort_values('Number of annotations', 0, False)
    sorted_temp_df = df.sort_index()
    # background = 0 에 해당되는 label 추가 후 기존들을 모두 label + 1 로 설정
    sorted_df = pd.DataFrame(["Backgroud"], columns=["Categories"])
    sorted_df = sorted_df.append(sorted_temp_df, ignore_index=True)
    category_names = list(sorted_df.Categories)
    return category_names


class CustomDataLoader(Dataset):
    """COCO format"""

    def __init__(self, data_dir, mode='train', transform=None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)
        self.category_names = get_category_names()
        # self.color_transform = color_transform

    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]

        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(
            dataset_path, image_infos['file_name']))
        # img_ycrcb = cv2.cvtColor(images, cv2.COLOR_BGR2YCrCb)
        # ycrcb_planes = cv2.split(img_ycrcb)

        # # histogram equalization
        # ycrcb_planes[0] = cv2.equalizeHist(ycrcb_planes[0])
        # dst_ycrcb = cv2.merge(ycrcb_planes)
        images = cv2.cvtColor(
            images, cv2.COLOR_BGR2RGB)
        # images /= 255.0

        if (self.mode in ('train', 'val')):
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
                pixel_value = self.category_names.index(className)
                masks = np.maximum(self.coco.annToMask(
                    anns[i])*pixel_value, masks)
            masks = masks.astype(np.float32)

            # transform -> albumentations 라이브러리 활용
            # if self.color_transform is not None:
            #     color_transformed = self.color_transform(image=images)
            #     images = color_transformed["image"]

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
