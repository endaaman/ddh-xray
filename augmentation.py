import os
import random
import enum
from typing import List

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
from matplotlib import pyplot as plt

from datasets import XrayDataset
from utils import Annotation, BBLabel, pil_to_tensor, tensor_to_pil


IMAGE_SIZE = 624

def float_level(level, maxval):
    return level * maxval

def int_level(level, maxval):
    return int(level * maxval)

class Augmentation():
    def __init__(self,
                 tile_size=512,
                 ways=3,
                 level_range=(0.2, 0.8),
                 scale_range=(0.875, 1.25)):
        self.tile_size = tile_size
        self.ways = ways
        self.names = [
            # 'autocontrast',
            'equalize',
            'posterize',
            'solarize',
            # 'color',
            # 'shear_x',
            # 'shear_y',
            'high_contrast',
            'low_contrast',
            'high_brightness',
            'low_brightness',
            # 'high_sharpness',
            # 'low_sharpness',
        ]
        self.level_range = level_range
        self.scale_range = scale_range

    # NOT USED: no effect
    def aug_autocontrast(self, img, _):
        return ImageOps.autocontrast(img)

    def aug_equalize(self, img, level):
        if level > 0.5:
            return ImageOps.equalize(img)
        return img

    def aug_posterize(self, img, level):
        i = 2 + int_level(1 - level, 2)
        # lowest should be 2
        return ImageOps.posterize(img, i)

    def aug_solarize(self, img, level):
        return ImageOps.solarize(img, int_level(1 - level, 100) + 156)

    # NOT USED: echo image is basically grayscale
    def aug_color(self, img, level):
        v = float_level(level, 1.8) + 0.1
        return ImageEnhance.Color(img).enhance(v)

    def aug_high_contrast(self, img, level):
        v = 1 + float_level(level, 3)
        return ImageEnhance.Contrast(img).enhance(v)

    def aug_low_contrast(self, img, level):
        v = 1 - float_level(level, 0.8)
        return ImageEnhance.Contrast(img).enhance(v)

    def aug_high_brightness(self, img, level):
        v = 1 + float_level(level, 2)
        return ImageEnhance.Brightness(img).enhance(v)

    def aug_low_brightness(self, img, level):
        v = 1 - float_level(level, 0.6)
        return ImageEnhance.Brightness(img).enhance(v)

    # NOT USED: no effect
    def aug_high_sharpness(self, img, level):
        v = 1 + float_level(level, 6)
        return ImageEnhance.Sharpness(img).enhance(v)

    # NOT USED: no effect
    def aug_low_sharpness(self, img, level):
        v = 1 - float_level(level, 0.98)
        return ImageEnhance.Sharpness(img).enhance(v)

    # NOT USED: causes invalid lavel
    def aug_shear_x(self, img, level):
        v = float_level(level, 0.3)
        if np.random.uniform() > 0.5:
            v = -v
        return img.transform((img.width, img.height),
                             Image.AFFINE, (1, v, 0, 0, 1, 0),
                             resample=Image.BILINEAR)

    # NOT USED: causes invalid lavel
    def aug_shear_y(self, img, level):
        v = float_level(level, 0.3)
        if np.random.uniform() > 0.5:
            v = -v
        return img.transform((img.width, img.height),
                             Image.AFFINE, (1, 0, 0, v, 1, 0),
                             resample=Image.BILINEAR)

    def aug(self, img, name=None, level=None):
        if not name:
            name = random.choice(self.names)
        if level == None:
            level = random.uniform(*self.level_range)
        process = getattr(self, f'aug_{name}')
        if not process:
            raise Exception(f'Invalid aug nema: {name}')
        mode = img.mode
        return process(img.convert('RGB'), level).convert(mode)

    def aug_multi(self, imgs, name=None, level=None):
        return [self.aug(img, name, level) for img in imgs]

    def resize_and_crop(self, img: Image, label: BBLabel):
        min_scale = self.scale_range[0]
        max_scale = self.scale_range[1]
        scale = np.random.rand() * (max_scale - min_scale) + min_scale
        new_size = int(IMAGE_SIZE * scale)
        img = img.resize((new_size, new_size))

        # extract rect
        rects = np.vstack([a.rect for a in label])
        rects *= scale

        t = self.tile_size
        x_offset = np.random.randint(0, new_size - t + 1)
        y_offset = np.random.randint(0, new_size - t + 1)
        # crop
        img = img.crop((x_offset, y_offset, x_offset + t, y_offset + t))
        # subtract offset in abs coord
        rects -= np.array([x_offset, y_offset, x_offset, y_offset])
        label = [Annotation(label[i].id, rect) for i, rect in enumerate(rects)]
        return img, label

    def multi_way_aug(self, img: Image, label: BBLabel, level=None):
        ws = np.random.dirichlet([1] * self.ways).astype(np.float32)
        tensors = []
        for i in range(self.ways):
            if not level:
                level = random.uniform(*self.level_range)
            tensor = pil_to_tensor(self.aug(img, level=level))
            tensors.append(tensor * ws[i])
        tensors = torch.stack(tensors)
        mixed = torch.sum(tensors, axis=0)
        return tensor_to_pil(mixed), label

    def __call__(self, img: Image, label: BBLabel, level=None):
        img, label = self.resize_and_crop(img, label)
        return self.multi_way_aug(img, label)

class ResizeAugmentation():
    def __init__(self, tile_size):
        self.tile_size = tile_size

    def __call__(self, img: Image, label: BBLabel):
        scale = self.tile_size / IMAGE_SIZE
        return img.resize((self.tile_size, self.tile_size)), [Annotation(a.id, a.rect *scale) for a in label]

if __name__ == '__main__':
    ds = XrayDataset()
    ds.set_augmentaion(Augmentation(tile_size=512))

    for i, (image, label) in enumerate(ds):
        draw = ImageDraw.Draw(image)

        for a in label:
            print(a)
            x0, y0, x1, y1 = a.rect
            draw.rectangle(((x0, y0), (x1, y1)), outline='yellow', width=1)
        image.save(f'tmp/{i}.jpg')
        if i > 2:
            break


