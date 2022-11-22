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

import pandas as pd
from sklearn.model_selection import train_test_split
from recordclass import recordclass, RecordClass
from tqdm import tqdm
import numpy as np
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

col_target = 'treatment'
cols_measure = ['left_alpha', 'right_alpha', 'left_oe', 'right_oe', 'left_a', 'right_a', 'left_b', 'right_b', ]

# cols_cat = ['female', 'family_history', 'breech_presentation', 'skin_laterality', 'limb_limitation']
# cols_cat = ['female', 'family_history', 'skin_laterality', 'limb_limitation']
# cols_val = ['left_alpha', 'right_alpha', 'left_oe', 'right_oe', 'left_a', 'right_a', 'left_b', 'right_b', ]

cols_cat = []
cols_val = ['female', 'breech_presentation'] + cols_measure
# cols_val = ['female', 'breech_presentation'] + cols_measure

do_abs = lambda x: np.power(x, 2)
# do_abs = lambda x: x
cols_extend = {
    # 'alpha_diff': lambda x: do_abs(x['left_alpha'] - x['right_alpha']),
    # 'oe_diff': lambda x: do_abs(x['left_oe'] - x['right_oe']),
    # 'a_diff': lambda x: do_abs(x['left_a'] - x['right_a']),
    # 'b_diff': lambda x: do_abs(x['left_b'] - x['right_b']),
}
cols_feature = cols_cat + cols_val + list(cols_extend.keys())

col_to_label = {
    'female': 'Female',
    'breech_presentation': 'Breech presentation',
    'left_a': 'Left A',
    'right_a': 'Right A',
    'left_b': 'Left B',
    'right_b': 'Right B',
    'left_alpha': 'Left α',
    'right_alpha': 'Right α',
    'left_oe': 'Left OE',
    'right_oe': 'Right OE',
}


ORIGINAL_IMAGE_SIZE = 624
IMAGE_MEAN = 0.4838
IMAGE_STD = 0.3271

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


XRBBItem = namedtuple('XRItem', ['image', 'name', 'bb', 'image_path', 'label_path'])


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



class ROIDataset(Dataset):
    def __init__(self, target='train', mode='default', test_ratio=0.25, image_size=512, normalized=True, seed=42):
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
        self.items = self.load_items()

        self.horizontal_filpper_index = None

        # mean, std = calc_mean_and_std([item.image for item in self.items])
        # print(mean, std)

        if self.target == 'train':
            augs = [
                A.RandomResizedCrop(width=image_size, height=image_size, scale=[0.7, 1.0]),
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

    def load_items_from_dir(self, di):
        pass

    def load_items(self):
        base_dir = 'data/roi'
        paths_all = sorted(glob(os.path.join(base_dir, 'image/*.jpg')))

        paths_train, paths_test = train_test_split(
            paths_all,
            test_size=self.test_ratio,
            random_state=self.seed)

        paths = paths_all
        if self.target == 'train':
            paths = paths_train
        elif self.target == 'test':
            paths = paths_test

        items = []
        t = tqdm(paths, leave=False)
        for image_path in t:
            file_name = os.path.basename(image_path)
            base_name = os.path.splitext(file_name)[0]

            label_path = os.path.join(base_dir, f'label/{base_name}.txt')
            bb_df = read_label_as_df(label_path)
            image = Image.open(image_path)
            items.append(XRBBItem(image, base_name, bb_df, image_path, label_path))
            t.set_description(f'loaded {image_path}')
            t.refresh()
        print('All images loaded')
        return items

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


class ROICroppedDataset(Dataset):
    def __init__(self, image_size=(256, 128), **kwargs):
        super().__init__(**kwargs)
        full_df = pd.read_excel('data/table.xlsx')
        self.df = full_df[full_df['test'] == 0 if self.is_training else 1]
        self.image_size = image_size
        # self.df = full_df

    def __getitem__(self, idx):
        item = self.items[idx]
        image = item.image
        vv = np.round(item.label.values[:, :4]).astype(np.int)
        # top-left edge
        left, top = np.min(vv[:, [0, 1]], axis=0)
        # bottom-right edge
        right, bottom = np.max(vv[:, [2, 3]], axis=0)
        x = image.crop((left, top, right, bottom))
        found = self.df[self.df['label'] == int(item.name)]

        y = int(found['treatment'].values[0])
        if self.augmentation:
            x, y = self.augmentation(x, y)
        # if y > 0:
        #     x = ImageOps.invert(x)
        return self.transform(x, y)


class C(Commander):
    def arg_common(self, parser):
        parser.add_argument('--target', '-t', default='all', choices=['all', 'train', 'test'])
        parser.add_argument('--mode', '-m', default='default', choices=['default', 'effdet', 'yolo', 'ssd'])

    def pre_common(self):
        self.ds = ROIDataset(
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
