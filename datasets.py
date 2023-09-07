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
from pydantic import Field

from endaaman.cli import BaseCLI
from endaaman.ml import pil_to_tensor, tensor_to_pil

from common import load_data, cols_clinical, cols_measure, col_target
from utils import draw_bb, draw_bbs


ImageFile.LOAD_TRUNCATED_IMAGES = True

LABEL_TO_STR = {
    1: 'right top',
    2: 'right out',
    3: 'right in',
    4: 'left top',
    5: 'left out',
    6: 'left in',
}

J = os.path.join


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


def read_label_as_df(path, size=ORIGINAL_IMAGE_SIZE):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cols = ['x0', 'y0', 'x1', 'y1', 'label']
    data = []
    for line in lines:
        parted  = line.split(' ')
        class_id = int(parted[0]) + 1
        center_x, center_y, w, h = [float(v) * size for v in parted[1:]]
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
    test: bool
    name: str
    image: ImageType
    clinical: np.ndarray
    measurement: np.ndarray
    treatment: bool

class BBItem(NamedTuple):
    test: bool
    name: str
    image: ImageType
    bb: pd.DataFrame


def pad_bb(bboxes, labels, size, fill=(0, 0)):
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
        bboxes, labels = pad_bb(bboxes, labels, 6)

        labels = {
            'bbox': torch.FloatTensor(bboxes),
            'cls': torch.FloatTensor(labels),
        }
        return images, labels

class YOLOAdapter():
    def __call__(self, images, bboxes, labels):
        bboxes = xyxy_to_xywh(bboxes, w=images.shape[2], h=images.shape[1])
        bboxes, labels = pad_bb(bboxes, labels, 6, fill=(0, -1))
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

# pylint: disable=abstract-method
class BaseDataset(Dataset):
    def __init__(self, target, basedir='data', aug_mode='same', normalize_image=True, normalize_features=True, seed=42):
        self.target = target
        self.basedir = basedir
        self.aug_mode = aug_mode
        self.normalize_image = normalize_image
        self.normalize_features = normalize_features
        self.seed = seed

        dfs = load_data(test_ratio=0, normalize_features=normalize_features, seed=seed)
        df = dfs['all']

        names = sorted(glob(J(basedir, 'images', '*.jpg')))
        labels = [os.path.splitext(os.path.basename(p))[0] for p in names]
        df = df[df.index.isin(labels)]
        self.df = df


class XRBBDataset(BaseDataset):
    def __init__(self, size:int, mode='default', **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.adapter = {
            'default': DefaultDetAdaper(),
            'effdet': EffDetAdaper(),
            'yolo': YOLOAdapter(),
            'ssd': SSDAdapter(),
        }[mode]

        self.items = []
        for idx, row in tqdm(self.df.iterrows(), leave=False, total=len(self.df)):
            image_path = f'data/images/{idx}.jpg'
            label_path = f'data/labels/{idx}.txt'
            self.items.append(BBItem(
                test=row.test,
                name=idx,
                image=Image.open(image_path),
                bb=read_label_as_df(label_path)))
        print(f'{self.target} images loaded')

        augss = {
            'train': [
                # A.CenterCrop(width=512, height=512),
                # A.Resize(width=size, height=size),
                # A.Rotate(limit=(-5, 5)),
                A.RandomResizedCrop(width=size, height=size, scale=[0.9, 1.1]),
                *BASE_AUGS,
                # A.HorizontalFlip(p=0.5),
            ],
            'test': [
                # A.CenterCrop(width=512, height=512),
                A.Resize(width=size, height=size),
            ]
        }
        augss['all'] = augss['test']
        augss['same'] = augss[self.target]
        augs = augss[self.aug_mode]

        self.albu = A.ReplayCompose([
            *augs,
            A.Normalize(*[[v] * 3 for v in [IMAGE_MEAN, IMAGE_STD]]) if self.normalize_image else None,
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

        self.horizontal_filpper_index = None
        for i, t in enumerate(self.albu):
            if isinstance(t, A.HorizontalFlip):
                self.horizontal_filpper_index = i
                break

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


class BaseImageDataset(BaseDataset):
    def __init__(self, num_features=0, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features

    def load_items(self, basedir):
        items = []
        for idx, row in tqdm(self.df.iterrows(), leave=False, total=len(self.df)):
            items.append(ClsItem(
                test=row.test,
                name=idx,
                image=Image.open(os.path.join(basedir, 'images', f'{idx}.jpg')).copy(),
                clinical=row[cols_clinical],
                measurement=row[cols_measure],
                treatment=row.treatment))
        print(f'Loaded {len(items)} images ({self.target}) from {basedir}')
        return items

    def __len__(self):
        return len(self.items)

    def create_augs(self, crop_size, width, height):
        common_augs = [
            A.CenterCrop(width=width, height=height),
        ] if crop_size > 0 else []
        augss = {
            'train': common_augs + [
                A.RandomResizedCrop(width=width, height=height, scale=[0.9, 1.1]),
                # A.RandomCrop(width=512, height=512),
                *BASE_AUGS,
            ],
            'test': common_augs + [
                A.Resize(width=width, height=height),
            ],
            'none': [],
        }
        augss['all'] = augss['test']
        augss['same'] = augss[self.target]

        augs = augss[self.aug_mode]

        if self.normalize_image:
            augs.append(A.Normalize(*[[v] * 3 for v in [IMAGE_MEAN, IMAGE_STD]]))
        augs.append(A.ToGray(p=1))
        augs.append(ToTensorV2())
        return augs

    def __getitem__(self, idx):
        item = self.items[idx]
        x = self.albu(image=np.array(item.image))['image']
        y = torch.FloatTensor([item.treatment])

        x = x.mean(dim=0)[None, ...]
        if self.num_features > 0:
            v = pd.concat([item.measurement, item.clinical]).values[:self.num_features]
            features = torch.from_numpy(v).to(torch.float)
            x = (x, features)
        return x, y

class FeatureDataset(BaseDataset):
    def __init__(self, num_features=8, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = row[cols_measure + cols_clinical].values[:self.num_features]
        y = row[col_target]
        x = torch.tensor(x).to(torch.float)
        y = torch.tensor(y)[None].to(torch.float)
        return x, y


class XRDataset(BaseImageDataset):
    def __init__(self, size=512, crop_size=-1, **kwargs):
        super().__init__(**kwargs)
        self.items = self.load_items(basedir=self.basedir)
        self.albu = A.Compose(self.create_augs(crop_size, size, size))


class XRROIDataset(BaseImageDataset):
    def __init__(self, size=512, crop_size=-1, **kwargs):
        super().__init__(**kwargs)
        self.items = self.load_items(basedir=self.basedir)
        # ignore crop_size
        self.albu = A.Compose(self.create_augs(-1, size, size//2))


class CLI(BaseCLI):
    class CommonArgs(BaseCLI.CommonArgs):
        target: str = Field('all', cli=('-t', ), regex=r'^all|train|test$')
        basedir: str = Field('data/all', cli=('-b', ))
        size:int = 512
        num_features: int = Field(0, cli=('-f', '--features'), )

    def run_cls(self, a:CommonArgs):
        self.ds = XRDataset(target=a.target, basedir=a.basedir, size=a.size, num_features=a.num_features)

    def run_roi(self, a:CommonArgs):
        self.ds = XRROIDataset(target=a.target, basedir=a.basedir, size=a.size, num_features=a.num_features)

    def run_feature(self, a:CommonArgs):
        self.ds = FeatureDataset(target=a.target, basedir=a.basedir, size=a.size, num_features=a.num_features)

    class BbArgs(CommonArgs):
        pass
        # mode: str = Field('default', regex='^default|effdet|yolo|ssd$')

    def run_bb(self, a:BbArgs):
        self.ds = XRBBDataset(target=a.target, size=a.size, mode='default')
        dest = f'tmp/bb_{a.target}'
        os.makedirs(dest, exist_ok=True)
        t = tqdm(enumerate(self.ds), total=len(self.ds))
        for i, (img, bb, label) in t:
            img = tensor_to_pil(img)
            bbs = torch.from_numpy(np.concatenate([bb, label[..., None]], axis=1))
            ret = draw_bbs([img], [bbs])
            p = f'{dest}/{i}.jpg'
            ret[0].save(p)
            t.set_description(p)
            t.refresh()


    def run_t(self):
        pass


if __name__ == '__main__':
    cli = CLI()
    cli.run()
