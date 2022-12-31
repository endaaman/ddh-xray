import os
import re
import shutil
from glob import glob
import warnings
import math
import time
import functools
from pprint import pprint
from collections import namedtuple, Counter
from typing import NamedTuple

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from PIL.Image import Image as ImageType
from PIL import Image, ImageFilter, ImageOps, ImageFile
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


from endaaman import Commander
from endaaman.torch import pil_to_tensor, tensor_to_pil

from utils import XrayBBItem, calc_mean_and_std, label_to_tensor, draw_bb

ImageFile.LOAD_TRUNCATED_IMAGES = True

ORIGINAL_IMAGE_SIZE = 624
IMAGE_MEAN = 0.4838
IMAGE_STD = 0.3271

BASE_AUGS = [
    A.HorizontalFlip(p=0.5),
    A.GaussNoise(p=0.2),
    A.OneOf([
        A.MotionBlur(p=.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=5, p=0.5),
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.Emboss(),
        A.RandomBrightnessContrast(),
    ], p=0.3),
    A.HueSaturationValue(p=0.3),
]

def read_label_as_df(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cols = ['x0', 'y0', 'x1', 'y1', 'id']
    data = []
    for line in lines:
        parted  = line.split(' ')
        class_id = int(parted[0]) + 1
        center_x, center_y, w, h = [float(v) * ORIGINAL_IMAGE_SIZE for v in parted[1:]]
        data.append([
            # convert yolo to pascal voc
            center_x - w / 2,
            center_y - h / 2,
            center_x + w / 2,
            center_y + h / 2,
            class_id,
        ])
    return pd.DataFrame(columns=cols, data=data)

def label_to_str(label):
    lines = []
    for bbox in label:
        x0, y0, x1, y1 = bbox[:4] / ORIGINAL_IMAGE_SIZE
        cls_ = bbox[4] - 1
        x = (x1 - x0) / 2
        y = (y1 - y0) / 2
        w = x1 - x0
        h = y1 - y0
        line = f'{cls_} {x:.6f} {y:.6f} {w:.6f} {h:.6f}'
        lines.append([cls_, line])
    lines = sorted(lines, key=lambda v:v[0])
    return '\n'.join([l[1] for l in lines])



class ClsItem(NamedTuple):
    name: str
    image: ImageType
    treatment: bool

class BBItem(NamedTuple):
    name: str
    image: ImageType
    bb: pd.DataFrame


def pad_to_fixed_size(img, size=(256, 128), bg=(0,0,0)):
    is_wide = img.width / img.height > 2
    bg = Image.new('RGB', size, bg)
    if is_wide:
        new_height = round(img.height * size[0] / img.width)
        img = img.resize((size[0], new_height))
        bg.paste(img, (0, (size[1] - new_height)//2))
    else:
        new_width = round(img.width * size[1] / img.height)
        img = img.resize((new_width, size[1]))
        bg.paste(img, ((size[0] - new_width)//2, 0))
    return bg

def pad_targets(bboxes, labels, size, fill=(0, 0)):
    pad_bboxes = np.zeros([size, 4], dtype=bboxes.dtype)
    pad_labels = np.zeros([size], dtype=bboxes.dtype)
    if fill:
        pad_bboxes.fill(fill[0])
        pad_labels.fill(fill[1])
    pad_bboxes[0:bboxes.shape[0], :] = bboxes
    pad_labels[0:labels.shape[0]] = labels
    return pad_bboxes, pad_labels

def xyxy_to_xywh(bb, w, h):
    ww = (bb[:, 2] - bb[:, 0])
    hh = (bb[:, 3] - bb[:, 1])
    xx = bb[:, 0] + ww * 0.5
    yy = bb[:, 1] + hh * 0.5
    bb = np.stack([xx, yy, ww, hh], axis=1)
    bb[:, [0, 2]] /= w
    bb[:, [1, 3]] /= h
    return bb


class DefaultDetAdaper():
    def __call__(self, images, bboxes, labels):
        return images, bboxes, labels

class EffDetAdaper():
    def __call__(self, images, bboxes, labels):
        # from xyxy to yxyx
        bboxes = bboxes[:, [1, 0, 3, 2]]
        bboxes, labels = pad_targets(bboxes, labels, 6)

        labels = {
            'bbox': torch.FloatTensor(bboxes),
            'cls': torch.FloatTensor(labels),
        }
        return images, labels

class YOLOAdapter():
    def __call__(self, images, bboxes, labels):
        bboxes = xyxy_to_xywh(bboxes, w=images.shape[2], h=images.shape[1])
        bboxes, labels = pad_targets(bboxes, labels, 6, fill=(0, -1))
        batches = np.zeros([6, 1])
        # yolo targets: [batch_idx, class_id, x, y, w, h]
        labels = np.concatenate([batches, labels[:, None], bboxes], axis=1)
        return images, torch.FloatTensor(labels)


class SSDAdapter():
    def __call__(self, images, bboxes, labels):
        # bboxes = xyxy_to_xywh(bboxes, w=images.shape[2], h=images.shape[1])
        if len(bboxes) > 0:
            bboxes[:, [0, 2]] /= images.shape[2]
            bboxes[:, [1, 3]] /= images.shape[1]
        # bboxes = [b for b in bboxes]
        # labels = [l for l in labels]
        return images, (torch.from_numpy(bboxes), torch.from_numpy(labels))


def load_df(target, test_ratio, seed):
    df_all = pd.read_excel('data/table.xlsx', index_col=0, converters={'label': str})

    if test_ratio > 0:
        df_train, df_test = train_test_split(
            df_all,
            test_size=test_ratio,
            random_state=seed,
            stratify=df_all['treatment'],
        )
        df_test['test'] = 1
        df_train['test'] = 0
    else:
        df_train = df_all[df_all['test'] > 0]
        df_test = df_all[df_all['test'] < 1]

    return {
        'all': df_all,
        'train': df_train,
        'test': df_test,
    }[target]


class XRBBDataset(Dataset):
    def __init__(self, target='train', mode='default', test_ratio=-1, size=512, normalized=True, seed=42):
        self.target = target
        self.mode = mode
        self.normalized = normalized
        self.seed = seed
        self.test_ratio = test_ratio

        self.adapter = {
            'default': DefaultDetAdaper(),
            'effdet': EffDetAdaper(),
            'yolo': YOLOAdapter(),
            'ssd': SSDAdapter(),
        }[mode]

        self.df = load_df(target, test_ratio, seed)
        self.items = []
        for idx in tqdm(self.df.index, leave=False, total=len(self.df)):
            image_path = f'data/images/{idx}.jpg'
            label_path = f'data/label/{idx}.txt'
            self.items.append(BBItem(
                name=idx,
                image=Image.open(image_path),
                bb=read_label_as_df(label_path)))
        print(f'{target} images loaded')

        self.horizontal_filpper_index = None
        # mean, std = calc_mean_and_std([item.image for item in self.items])
        # print(mean, std)

        if self.target == 'train':
            augs = [
                A.RandomResizedCrop(width=size, height=size, scale=[0.7, 1.0]),
                *BASE_AUGS
            ]
        else:
            augs = []

        self.albu = A.ReplayCompose([
            *augs,
            A.Normalize(*[[v] * 3 for v in [IMAGE_MEAN, IMAGE_STD]]) if self.normalized else None,
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

        self.horizontal_filpper_index = None
        for i, t in enumerate(self.albu):
            if isinstance(t, A.HorizontalFlip):
                self.horizontal_filpper_index = i
                break

    def validate_id(self):
        for item in tqdm(self.items):
            m = np.all(item.bb['id'].values == np.array([1, 2, 3, 4, 5, 6]))
            if not m:
                print(item.name)
                print(item.bb)
                print()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = self.albu(
                image=np.array(item.image),
                bboxes=item.bb.values[:, :4],
                labels=item.bb.values[:, 4],
            )
        images = result['image']
        bboxes = np.array(result['bboxes'])
        labels = np.array(result['labels'])
        if self.horizontal_filpper_index is not None:
            flipped = result['replay']['transforms'][self.horizontal_filpper_index]['applied']
            if flipped:
                labels -= 3
                labels[labels < 1] += 6

        return self.adapter(images, bboxes, labels)



class XRDataset(Dataset):
    def __init__(self, target='train', test_ratio=-1, size=512, normalized=True, seed=42):
        self.target = target
        self.test_ratio = test_ratio
        self.normalized = normalized
        self.seed = seed

        self.df = load_df(target, test_ratio, seed)
        self.items = []
        for idx, row in tqdm(self.df.iterrows(), leave=False, total=len(self.df)):
            self.items.append(ClsItem(
                name=idx,
                image=Image.open(f'data/images/{idx}.jpg'),
                treatment=row.treatment))
        print(f'{target} images loaded')

        if self.target == 'train':
            augs = [
                A.RandomResizedCrop(width=size, height=size, scale=[0.7, 1.0]),
                *BASE_AUGS
            ]
        else:
            augs = []

        if normalized:
            augs.append(A.Normalize(*[[v] * 3 for v in [IMAGE_MEAN, IMAGE_STD]]))
        augs.append(ToTensorV2())
        self.albu = A.Compose(augs)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        x = self.albu(image=np.array(item.image))['image']
        y = torch.FloatTensor([item.diagnosis])
        return x, y


class XRROIDataset(Dataset):
    def __init__(self, base_dir='data/roi', target='train', test_ratio=-1, size=(512, 256), normalized=True, seed=42):
        self.size = size

        self.target = target
        self.test_ratio = test_ratio
        self.normalized = normalized
        self.seed = seed

        self.df = load_df(target, test_ratio, seed)
        self.items = []
        for idx, row in tqdm(self.df.index, leave=False, total=len(self.df)):
            self.items.append(ClsItem(
                name=idx,
                image=Image.open(os.path.join(base_dir, f'{idx}.jpg')),
                treatment=row.treatment))
        print(f'{target} images loaded')

        if self.target == 'train':
            augs = [
                A.RandomResizedCrop(width=size[0], height=size[1], scale=[0.7, 1.0]),
                *BASE_AUGS
            ]
        else:
            augs = []

        if normalized:
            augs.append(A.Normalize(*[[v] * 3 for v in [IMAGE_MEAN, IMAGE_STD]]))
        augs.append(ToTensorV2())
        self.albu = A.Compose(augs)


    def __getitem__(self, idx):
        item = self.items[idx]
        x = self.albu(image=np.array(item.image))['image']
        y = torch.FloatTensor([item.diagnosis])
        return x, y

    # def __getitem__(self, idx):
    #     item = self.items[idx]
    #     image = item.image
    #     vv = np.round(item.label.values[:, :4]).astype(np.int)
    #     # top-left edge
    #     left, top = np.min(vv[:, [0, 1]], axis=0)
    #     # bottom-right edge
    #     right, bottom = np.max(vv[:, [2, 3]], axis=0)
    #     x = image.crop((left, top, right, bottom))
    #     found = self.df[self.df['label'] == int(item.name)]
    #
    #     y = int(found['treatment'].values[0])
    #     if self.augmentation:
    #         x, y = self.augmentation(x, y)
    #     # if y > 0:
    #     #     x = ImageOps.invert(x)
    #     return self.transform(x, y)


class C(Commander):
    def arg_common(self, parser):
        parser.add_argument('--target', '-t', default='all', choices=['all', 'train', 'test'])
        parser.add_argument('--mode', '-m', default='default', choices=['default', 'effdet', 'yolo', 'ssd'])

    def pre_common(self):
        self.ds = XRBBDataset(
            target=self.args.target,
            mode=self.args.mode,
        )

    def run_bbs(self):
        dest = f'out/bbs_{self.ds.target}'
        os.makedirs(dest, exist_ok=True)
        for i, (img, bb, label) in tqdm(enumerate(self.ds), total=len(self.ds)):
            # self.img, self.bb, self.label = item
            img = tensor_to_pil(img)
            ret = draw_bb(img, bb, {i:f'{l}' for i, l in enumerate(label) })
            ret.save(f'{dest}/{i}.jpg')


if __name__ == '__main__':
    c = C()
    c.run()
