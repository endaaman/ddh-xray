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
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils import XrayBBItem, calc_mean_and_std, label_to_tensor


class BaseDataset(Dataset):
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
            if self.is_training:
                file_name = os.path.basename(image_path)
                base_name = os.path.splitext(file_name)[0]
                label_path = os.path.join(base_dir, 'label', f'{base_name}.txt')
            else:
                label_path = None
            items.append(XrayBBItem(image_path, label_path))
            t.set_description(f'loaded {image_path}')
            t.refresh()
        print('All images loaded')
        return items

    def set_augmentaion(self, augmentation):
        self.augmentation = augmentation

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
        pass


class XrayDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        item = self.items[idx]
        x = item.image.copy()
        y = item.label.copy()
        if self.augmentation:
            x, y = self.augmentation(x, y)
        return self.transform(x, y)


class ROIDataset(BaseDataset):
    def __init__(self, image_size=(256, 128), *args, **kwargs):
        super().__init__(*args, **kwargs)
        full_df = pd.read_excel('data/table.xlsx')
        self.df = full_df[full_df['test'] == 0 if self.is_training else 1]
        self.image_size = image_size
        self.padding_bg = (0, 0, 0)
        # self.df = full_df

    def pad_to_fixed_size(self, img):
        is_wide = img.width / img.height > 2
        bg = Image.new('RGB', self.image_size, self.padding_bg)
        if is_wide:
            new_height = round(img.height * self.image_size[0] / img.width)
            img = img.resize((self.image_size[0], new_height))
            bg.paste(img, (0, (self.image_size[1] - new_height)//2))
        else:
            new_width = round(img.width * self.image_size[1] / img.height)
            img = img.resize((new_width, self.image_size[1]))
            bg.paste(img, ((self.image_size[0] - new_width)//2, 0))
        return bg

    def __getitem__(self, idx):
        item = self.items[idx]
        image = item.image.copy()
        vv = np.round(item.label.values[:, :4]).astype(np.int)
        left, top = np.min(vv[:, [0, 1]], axis=0)
        right, bottom = np.max(vv[:, [2, 3]], axis=0)
        roi = image.crop((left, top, right, bottom))
        found = self.df[self.df['label'] == int(item.name)]

        x = self.pad_to_fixed_size(roi)
        y = int(found['treatment'].values[0])
        if self.augmentation:
            x, y = self.augmentation(x, y)
        return self.transform(x, y)

if __name__ == '__main__':
    from augmentation import Augmentation, ResizeAugmentation
    ds = ROIDataset()

    t_x = transforms.Compose([
        transforms.ToTensor(),
        # ds.get_normalizer(),
        # lambda x: x.permute([0, 2, 1]),
    ])
    t_y = lambda y: torch.tensor(y, dtype=torch.float32)
    ds.set_transforms(t_x, t_y)
    loader = DataLoader(ds, batch_size=3, num_workers=1)
    for i, (x, y) in enumerate(loader):
        print(i)
