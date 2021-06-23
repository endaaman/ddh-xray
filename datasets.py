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



class Annotation(RecordClass):
    id: int
    x0: float
    y0: float
    x1: float
    y1: float

class Item(RecordClass):
    image: object # Image
    annot: Annotation


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

def read_annotation(path):
    f = open(path, 'r')
    lines = f.readlines()
    annot = []
    for line in lines:
        parted  = line.split(' ')
        id = int(parted[0])
        x, y, w, h = [float(v) for v in parted[1:]]
        annot.append(Annotation(id, x, y, x + w, y + h))
    return annot

class XrayDataset(BaseDataset):
    def __init__(self, is_training=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_training = is_training
        self.items = self.load_items()

    def load_items(self):
        if self.is_training:
            base_dir = 'data/yolo/train'
        else:
            base_dir = 'data/yolo/test'

        items = []
        paths = sorted(glob.glob(os.path.join(base_dir, 'image', '*.jpg')))
        # t = tqdm(paths)
        for path in paths:
            img = Image.open(path)
            if self.is_training:
                file_name = os.path.basename(path)
                base_name = os.path.splitext(file_name)[0]
                annot_path = os.path.join(base_dir, 'label', f'{base_name}.txt')
                annot = read_annotation(annot_path)
            else:
                annot = None
            items.append(Item(img, annot))
            # t.set_description(f'loaded {path}')
            # t.refresh()

        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return self.transform_x(item.image), self.transform_y(item.annot)


if __name__ == '__main__':
    ds = XrayDataset()

    for (x, y) in ds:
        print(x)
        print(y)
        break
