import os
import re
import glob
import math
import time
import functools
from pprint import pprint
import pandas as pd
from collections import namedtuple, Counter

from recordclass import recordclass, RecordClass
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A

from utils import XrayBBItem, calc_mean_and_std, label_to_tensor, draw_bb, tensor_to_pil

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
        class_id = class_id % 3
        data.append([
            center_x - w / 2,
            center_y - h / 2,
            center_x + w / 2,
            center_y + h / 2,
            class_id,
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
    def __init__(self, is_training=True, transform_x=None, transform_y=None):
        self.set_transforms(transform_x, transform_y)
        self.augmentation = None
        self.is_training = is_training
        self.items = self.load_items()

    def get_mean_and_std(self):
        return calc_mean_and_std([item.image for item in self.items])

    def get_normalizer(self):
        return transforms.Normalize(*[[v] * 3 for v in self.get_mean_and_std()])

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

    def set_albu(self, albu):
        self.albu = albu

    def set_transforms(self, transform_x, transform_y):
        self._transform_x = transform_x
        self._transform_y = transform_y

    def transform_x(self, x):
        if self._transform_x:
            return self._transform_x(x)
        else:
            return x

    def transform_y(self, y):
        if self._transform_y:
            return self._transform_y(y)
        else:
            return y

    def transform(self, x, y):
        return self.transform_x(x), self.transform_y(y)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        if self.albu:
            auged = self.albu(
                image=np.array(item.image),
                bboxes=item.bb.values,
                labels=np.zeros(item.bb.shape[0]),
            )
            x, y = Image.fromarray(auged['image']), np.array(auged['bboxes'])
        else:
            x, y = item.image, item.bb.values
        return self.transform(x, y)


class ROICroppedDataset(Dataset):
    def __init__(self, image_size=(256, 128), *args, **kwargs):
        super().__init__(*args, **kwargs)
        full_df = pd.read_excel('data/table.xlsx')
        self.df = full_df[full_df['test'] == 0 if self.is_training else 1]
        self.image_size = image_size
        # self.df = full_df

    def __getitem__(self, idx):
        item = self.items[idx]
        image = item.image.copy()
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

if __name__ == '__main__':
    transform = A.Compose([
        A.RandomCrop(width=400, height=400),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    ds = XRBBDataset()
    ds.set_albu(transform)
    for i, (x, y) in enumerate(ds):
        img = draw_bb(x, y[:, :4], [str(v) for v in y[:, 4]])
        img.save(f'tmp/{i}.png')
        if i > 10:
            break


    # t_x = transforms.Compose([
    #     transforms.ToTensor(),
    #     # ds.get_normalizer(),
    #     # lambda x: x.permute([0, 2, 1]),
    # ])
    # t_y = lambda y: torch.tensor(y, dtype=torch.float32)
    # ds.set_transforms(t_x, t_y)
    # loader = DataLoader(ds, batch_size=3, num_workers=1)
    # for i, (x, y) in enumerate(loader):
    #     print(i)
