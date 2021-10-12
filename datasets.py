import os
import re
import glob
import warnings
import math
import time
import functools
from pprint import pprint
import pandas as pd
from collections import namedtuple, Counter

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

from utils import XrayBBItem, calc_mean_and_std, label_to_tensor, draw_bb, tensor_to_pil

ImageFile.LOAD_TRUNCATED_IMAGES = True

col_target = 'treatment'
cols_measure = ['left_alpha', 'right_alpha', 'left_oe', 'right_oe', 'left_a', 'right_a', 'left_b', 'right_b', ]

# cols_cat = ['sex', 'family_history', 'breech_presentation', 'skin_laterality', 'limb_limitation']
# cols_cat = ['sex', 'family_history', 'skin_laterality', 'limb_limitation']
# cols_val = ['left_alpha', 'right_alpha', 'left_oe', 'right_oe', 'left_a', 'right_a', 'left_b', 'right_b', ]

cols_cat = []
cols_val = ['sex', 'breech_presentation'] + cols_measure
# cols_val = ['sex', 'breech_presentation'] + cols_measure

do_abs = lambda x: np.power(x, 2)
# do_abs = lambda x: x
cols_extend = {
    # 'alpha_diff': lambda x: do_abs(x['left_alpha'] - x['right_alpha']),
    # 'oe_diff': lambda x: do_abs(x['left_oe'] - x['right_oe']),
    # 'a_diff': lambda x: do_abs(x['left_a'] - x['right_a']),
    # 'b_diff': lambda x: do_abs(x['left_b'] - x['right_b']),
}
cols_feature = cols_cat + cols_val + list(cols_extend.keys())


IMAGE_SIZE = 624

def read_label(path):
    f = open(path, 'r')
    lines = f.readlines()

    cols = ['x0', 'y0', 'x1', 'y1', 'id']
    data = []
    for line in lines:
        parted  = line.split(' ')
        class_id = int(parted[0])
        # convert from yolo to pascal voc
        center_x, center_y, w, h = [float(v) * IMAGE_SIZE for v in parted[1:]]
        # class_id = class_id % 3
        data.append([
            center_x - w / 2,
            center_y - h / 2,
            center_x + w / 2,
            center_y + h / 2,
            class_id + 1,
        ])
    return pd.DataFrame(columns=cols, data=data)


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


class XRBBDataset(Dataset):
    def __init__(self, is_training=True, use_yxyx=True, normalized=True):
        self.augmentation = None
        self.is_training = is_training
        self.use_yxyx = use_yxyx
        self.normalized = normalized
        self.items = self.load_items()
        self.apply_augs([])
        self.horizontal_filpper_index = None

    def get_mean_and_std(self):
        return calc_mean_and_std([item.image for item in self.items])

    def load_items(self):
        if self.is_training:
            base_dir = 'data/yolo/train'
        else:
            base_dir = 'data/yolo/test'

        items = []
        image_paths = sorted(glob.glob(os.path.join(base_dir, 'image', '*.jpg')))
        t = tqdm(image_paths, leave=False)
        for image_path in t:
            file_name = os.path.basename(image_path)
            base_name = os.path.splitext(file_name)[0]
            if self.is_training:
                label_path = os.path.join(base_dir, 'label', f'{base_name}.txt')
                bb_df = read_label(label_path)
            else:
                label_path = None
                bb_df = pd.DataFrame()
            image = Image.open(image_path)
            items.append(XRBBItem(image, base_name, bb_df, image_path, label_path))
            t.set_description(f'loaded {image_path}')
            t.refresh()
        print('All images loaded')
        return items

    def apply_augs(self, augs):
        self.albu = A.ReplayCompose([
            *augs,
            A.Normalize(*[[v] * 3 for v in self.get_mean_and_std()]) if self.normalized else None,
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
        x = result['image']
        boxes = np.array(result['bboxes'])
        labels = np.array(result['labels'])

        if self.horizontal_filpper_index is not None:
            flipped = result['replay']['transforms'][self.horizontal_filpper_index]['applied']
            if flipped:
                labels -= 3
                labels[labels < 1] += 6

        if self.use_yxyx:
            boxes = boxes[:, [1, 0, 3, 2]]

        if boxes.shape[0] == 0:
            boxes = np.zeros([6, 4], dtype=boxes.dtype)
            labels = np.zeros([6], dtype=boxes.dtype)
        if boxes.shape[0] < 6:
            z = np.zeros([6 - boxes.shape[0], 4], dtype=boxes.dtype)
            boxes = np.concatenate([boxes, z])
            z = np.zeros([6 - labels.shape[0]], dtype=labels.dtype)
            labels = np.concatenate([labels, z])

        y = {
            'bbox': torch.FloatTensor(boxes),
            'cls': torch.FloatTensor(labels),
        }
        return x, y


class ROICroppedDataset(Dataset):
    def __init__(self, image_size=(256, 128), *args, **kwargs):
        super().__init__(*args, **kwargs)
        full_df = pd.read_excel('data/table.xlsx')
        self.df = full_df[full_df['test'] == 0 if self.is_training else 1]
        self.image_size = image_size
        # self.df = full_df

    def __getitem__(self, idx):
        item = self.items[idx]
        image = item.image
        vv = np.round(item.label.values[:, :4]).astype(np.int)
        left, top = np.min(vv[:, [0, 1]], axis=0)
        right, bottom = np.max(vv[:, [2, 3]], axis=0)
        roi = image.crop((left, top, right, bottom))
        found = self.df[self.df['label'] == int(item.name)]

        y = int(found['treatment'].values[0])
        if self.augmentation:
            x, y = self.augmentation(x, y)
        # if y > 0:
        #     x = ImageOps.invert(x)
        return self.transform(x, y)


def test_flip():
    a = A.ReplayCompose([
        A.HorizontalFlip(p=0.5)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    img = np.ones([30, 30, 3])
    label = np.array([
        [5, 5, 6, 6, 1]
    ])

    r = a(
        image=img,
        bboxes=label[:, :4],
        labels=label[:, 4],
    )
    for t in r['replay']['transforms']:
        print(t['applied'])

    print(r['bboxes'])
    print(r['labels'])

if __name__ == '__main__':
    # test_flip()
    # exit(0)

    augs = [
        # A.RandomCrop(width=100, height=100),
        A.RandomResizedCrop(width=512, height=512, scale=[0.1, 0.2]),
        A.HorizontalFlip(p=0.5),
        A.GaussNoise(p=0.2),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=5, p=0.5),
        # A.PiecewiseAffine(p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Emboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
    ]

    ds = XRBBDataset(use_yxyx=False, normalized=False)
    ds.apply_augs(augs)
    # for i, (x, y) in enumerate(ds):
    #     print(y)
    #     break
    #     if i > 20:
    #         break

    loader = DataLoader(
        ds,
        batch_size=24,
        num_workers=16,
    )
    for i in range(100):
        for (x, y) in tqdm(loader):
            assert(y['bbox'].shape[:-1] == y['cls'].shape)
            assert(y['bbox'].shape[-1] == 4)
        print(f'ok {i}')
    exit(0)

    # for i, (x, y) in enumerate(ds):
    #     t = draw_bounding_boxes(image=x, boxes=y['bbox'], labels=[str(v.item()) for v in y['cls']])
    #     # img = draw_bb(tensor_to_pil(x), y['bbox'], [str(v) for v in y['cls']])
    #     img = tensor_to_pil(t)
    #     print(i)
    #     print(y['bbox'].shape)
    #     print(y['cls'].shape)
    #     print(x.shape)
    #     img.save(f'tmp/augs/{i}.png')
    #     if i > 20:
    #         break
    for i, (x, y) in tqdm(enumerate(ds), total=len(ds)):
        t = draw_bounding_boxes(image=x, boxes=y['bbox'], labels=[str(v.item()) for v in y['cls']])
        # img = draw_bb(tensor_to_pil(x), y['bbox'], [str(v) for v in y['cls']])
        img = tensor_to_pil(t)
        img.save(f'tmp/aug/{i}.png')
        if i > 20:
            break
