import os
import re
import shutil
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
    with open(path, 'r') as f:
        lines = f.readlines()

    cols = ['x0', 'y0', 'x1', 'y1', 'id']
    data = []
    for line in lines:
        parted  = line.split(' ')
        class_id = int(parted[0]) + 1
        center_x, center_y, w, h = [float(v) * IMAGE_SIZE for v in parted[1:]]
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
        x0, y0, x1, y1 = bbox[:4] / IMAGE_SIZE
        c = bbox[4] - 1
        x = (x1 - x0) / 2
        y = (y1 - y0) / 2
        w = x1 - x0
        h = y1 - y0
        line = f'{c} {x:.6f} {y:.6f} {w:.6f} {y:.6f}'
        lines.append([c, line])
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


class ROIDataset(Dataset):
    def __init__(self, target='effdet', is_training=True, normalized=True):
        adapter_table = {
            'effdet': self.effdet_adapter,
            'yolo': self.yolo_adapter,
            'ssd': self.ssd_adapter,
        }
        if target not in list(adapter_table.keys()):
            raise ValueError(f'Invalid target: {target}')
        self.target = target
        self.is_training = is_training
        self.normalized = normalized

        self.adapter = adapter_table[target]
        self.items = self.load_items()
        self.apply_augs([])
        self.horizontal_filpper_index = None

    def load_items(self):
        if self.is_training:
            base_dir = 'data/roi/train'
        else:
            base_dir = 'data/roi/test'

        items = []
        image_paths = sorted(glob.glob(os.path.join(base_dir, 'image', '*.jpg')))
        t = tqdm(image_paths, leave=False)
        for image_path in t:
            file_name = os.path.basename(image_path)
            base_name = os.path.splitext(file_name)[0]
            label_path = os.path.join(base_dir, 'label', f'{base_name}.txt')
            bb_df = read_label(label_path)
            # if self.is_training:
            #     label_path = os.path.join(base_dir, 'label', f'{base_name}.txt')
            #     bb_df = read_label(label_path)
            # else:
            #     label_path = None
            #     bb_df = pd.DataFrame()
            image = Image.open(image_path)
            items.append(XRBBItem(image, base_name, bb_df, image_path, label_path))
            t.set_description(f'loaded {image_path}')
            t.refresh()
        print('All images loaded')
        return items

    def apply_augs(self, augs):
        mean, std = calc_mean_and_std([item.image for item in self.items])
        self.albu = A.ReplayCompose([
            *augs,
            A.Normalize(*[[v] * 3 for v in [mean, std]]) if self.normalized else None,
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

        self.horizontal_filpper_index = None
        for i, t in enumerate(self.albu):
            if isinstance(t, A.HorizontalFlip):
                self.horizontal_filpper_index = i
                break

    def validate(self):
        for item in tqdm(self.items):
            m = np.all(item.bb['id'].values == np.array([1, 2, 3, 4, 5, 6]))
            if not m:
                print(item.name)
                print(item.bb)
                print()

    def __len__(self):
        return len(self.items)

    def effdet_adapter(self, images, bboxes, labels):
        # use yxyx
        bboxes = bboxes[:, [1, 0, 3, 2]]
        bboxes, labels = pad_targets(bboxes, labels, 6)

        labels = {
            'bbox': torch.FloatTensor(bboxes),
            'cls': torch.FloatTensor(labels),
        }
        return images, labels

    def yolo_adapter(self, images, bboxes, labels):
        bboxes = xyxy_to_xywh(bboxes, w=images.shape[2], h=images.shape[1])
        bboxes, labels = pad_targets(bboxes, labels, 6, fill=(0, -1))
        batches = np.zeros([6, 1])
        # yolo targets: [batch_idx, class_id, x, y, w, h]
        labels = np.concatenate([batches, labels[:, None], bboxes], axis=1)
        return images, torch.FloatTensor(labels)

    def ssd_adapter(self, images, bboxes, labels):
        # bboxes = xyxy_to_xywh(bboxes, w=images.shape[2], h=images.shape[1])
        if len(bboxes) > 0:
            bboxes[:, [0, 2]] /= images.shape[2]
            bboxes[:, [1, 3]] /= images.shape[1]
        # bboxes = [b for b in bboxes]
        # labels = [l for l in labels]
        return images, (torch.from_numpy(bboxes), torch.from_numpy(labels))

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


def resplit():
    train_ds = ROIDataset(is_training=True)
    test_ds = ROIDataset(is_training=False)

    train_ids = np.loadtxt('./train.tsv', delimiter='\t').astype(np.int64).tolist()
    test_ids = np.loadtxt('./test.tsv', delimiter='\t').astype(np.int64).tolist()

    items = train_ds.items + test_ds.items


    dest_dir = 'data/roi_new'
    for a in ['train', 'test']:
        for b in ['image', 'label']:
            os.makedirs(os.path.join(dest_dir, a, b), exist_ok=True)

    for item in tqdm(items):
        i = int(item.name)
        if i in train_ids:
            train_ids.pop(train_ids.index(i))
            target = 'train'
        elif i in test_ids:
            test_ids.pop(test_ids.index(i))
            target = 'test'
        else:
            print(f'missing: {i}')

        label_name = os.path.basename(item.label_path)
        image_name = os.path.basename(item.image_path)
        shutil.copyfile(item.image_path, os.path.join(dest_dir, target, 'image', image_name))
        shutil.copyfile(item.label_path, os.path.join(dest_dir, target, 'label', label_name))

    print(train_ids)
    print(test_ids)




if __name__ == '__main__':
    resplit()
