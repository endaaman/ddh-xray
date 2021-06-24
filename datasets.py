import os
import re
import glob
import math
import time
import functools
from pprint import pprint
from collections import namedtuple, Counter

from recordclass import recordclass, RecordClass
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils import XrayBBItem, Annotation, BBLabel, calc_mean_and_std
from augmentation import Augmentation


class BaseDataset(Dataset):
    def __init__(self, transform_x=None, transform_y=None):
        self.set_transforms(transform_x, transform_y)
        self.augmentation = None

    def get_mean_and_std(self):
        return calc_mean_and_std([item.image for item in self.items])

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
        return 0

    def __getitem__(self, idx):
        pass


class XrayDataset(BaseDataset):
    def __init__(self, is_training=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_training = is_training
        self.items = self.load_items()

    def get_mean_and_std(self):
        return calc_mean_and_std([item.image for item in self.items])

    def load_items(self):
        if self.is_training:
            base_dir = 'data/yolo/train'
        else:
            base_dir = 'data/yolo/test'

        items = []
        image_paths = sorted(glob.glob(os.path.join(base_dir, 'image', '*.jpg')))
        t = tqdm(image_paths)
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

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        x = item.image
        y = item.label
        if self.augmentation:
            x, y = self.augmentation(x, y.copy())
        return self.transform(x, y)


if __name__ == '__main__':
    ds = XrayDataset()

    # transform_x = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # ])
    # transform_y = transforms.Compose([
    #     lambda aa: [[a.id, *a.rect] for a in aa],
    #     lambda y: torch.tensor(y, dtype=torch.float),
    # ])
    # ds.set_transforms(transform_x, transform_y)
    # ds.set_augmentaion(Augmentation())


    for (x, y) in ds:
        print(x.width, x.height)
        # print(y[0])
