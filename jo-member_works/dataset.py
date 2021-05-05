import os
import json
import cv2
import pandas as pd
import numpy as np
import torch.utils.data as data
from pycocotools.coco import COCO
from albumentations.pytorch import ToTensorV2
import albumentations as A

dataset_path = '../input/data'
anns_file_path = dataset_path + '/' + 'train.json'

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

cat_histogram = np.zeros(nr_cats, dtype=int)
for ann in anns:
    cat_histogram[ann['category_id']] += 1

df = pd.DataFrame({'Categories': cat_names, 'Number of annotations': cat_histogram})
df = df.sort_values('Number of annotations', 0, False)
sorted_temp_df = df.sort_index()

# background = 0 에 해당되는 label 추가 후 기존들을 모두 label + 1 로 설정
sorted_df = pd.DataFrame(["Backgroud"], columns=["Categories"])
sorted_df = sorted_df.append(sorted_temp_df, ignore_index=True)
category_names = list(sorted_df.Categories)


class NormalizedAugmentation:
    '''
    기본값인 mean 과 std를 사용한 Augmentation
    '''

    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25), **args):
        self.transform = A.Compose([
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ])

    def __call__(self, image):
        return self.transform(image)

class NewAugmentation:
    '''
    기본값인 mean 과 std를 사용한 Augmentation
    '''

    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25), **args):
        self.transform = A.Compose([
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ])
        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.Rotate(p=0.3, limit=45),
            A.Cutout(num_holes=4, max_h_size=20, max_w_size=20),
            A.CLAHE(),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2()
        ])



    def __call__(self, image,mode):
        if mode=='train':
            return self.train_transform(image)
        elif mode=='val':
            return self.transform(image)
        elif mode=='test':
            return self.transform(image)

class TrashDataset(data.Dataset):
    def __init__(self, data_dir, mode='train'):
        self.transform = None
        self.mode = mode
        self.data_dir = data_dir
        self.coco = COCO(self.data_dir)

    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            annss = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id + 1" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # Unknown = 1, General trash = 2, ... , Cigarette = 11
            for i in range(len(annss)):
                className = self.get_classname(annss[i]['category_id'], cats)
                pixel_value = category_names.index(className)
                masks = np.maximum(self.coco.annToMask(annss[i]) * pixel_value, masks)
            masks = masks

            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                if self.mode=='train':
                    transformed = self.transform.train_transform(image=images, mask=masks)
                    images = transformed["image"]
                    masks = transformed["mask"]
                elif self.mode=='val':
                    transformed = self.transform.transform(image=images, mask=masks)
                    images = transformed["image"]
                    masks = transformed["mask"]

            return images, masks, image_infos

        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform.transform(image=images)
                images = transformed["image"]

            return images, image_infos

    def get_classname(self, classID, cats):
        for i in range(len(cats)):
            if cats[i]['id'] == classID:
                return cats[i]['name']
        return "None"

    def set_transform(self, transform):
        '''
        :param transform: 우리가 원하는 transform으로 dataset의 transform을 설정해준다
        '''
        self.transform = transform

    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())
