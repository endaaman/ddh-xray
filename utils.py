from typing import Union, Optional, List, Tuple, Text, BinaryIO
import os

import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes


IMAGE_SIZE = 624

# class Annotation(RecordClass):
#     id: int
#     rect: np.ndarray
#
#     def copy(self):
#         return Annotation(self.id, self.rect.copy())
#
#     def to_tensor(self):
#         return torch.tensor([*a.rect, a.id], dtype=torch.float)
#
# class BBLabel(List[Annotation]):
#     def copy(self):
#         return [a.copy() for a in self]
#
#     def to_tensor(self):
#         return torch.vstack([a.to_tensor() for a in self])

def read_label(path):
    f = open(path, 'r')
    lines = f.readlines()

    cols = ['x0', 'y0', 'x1', 'y1', 'id']
    data = []
    for line in lines:
        parted  = line.split(' ')
        class_id = int(parted[0])
        center_x, center_y, w, h = [float(v) * IMAGE_SIZE for v in parted[1:]]
        data.append([
            center_x - w / 2,
            center_y - h / 2,
            center_x + w / 2,
            center_y + h / 2,
            class_id,
        ])
    return pd.DataFrame(columns=cols, data=data)

def draw_bb(img:Image, bb:np.ndarray, labels:List[str]=None):
    # img = torch.from_numpy(x).permute(2, 0, 1)
    img = (pil_to_tensor(img) * 255).type(torch.uint8)
    t = draw_bounding_boxes(image=img, boxes=torch.from_numpy(bb), labels=labels)
    return tensor_to_pil(t)


def draw_bbs(imgs, bbss, color='yellow'):
    font = ImageFont.truetype('/usr/share/fonts/ubuntu/Ubuntu-L.ttf', 20)
    results = []
    for img, bbs in zip(imgs, bbss):
        draw = ImageDraw.Draw(img)
        for _, bb in enumerate(bbs):
            label = bb[4].long().item()
            bbox = bb[:4]
            draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline=color, width=1)
            # draw.text((bbox[0], bbox[1]), LABEL_TO_STR[label], font=font, fill=color)
        results.append(img)
    return results


def get_state_dict(model):
    if type(model) == torch.nn.DataParallel:
        return model.module.state_dict()
    else:
        return model.state_dict()

def label_to_tensor(label):
    return torch.from_numpy(label.values).type(torch.FloatTensor)

class XrayBBItem:
    def __init__(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path
        self.name = os.path.splitext(os.path.basename(image_path))[0]
        self.image = Image.open(self.image_path)
        if self.image.width != IMAGE_SIZE or self.image.height != IMAGE_SIZE:
            raise Exception(f'invalid size! {image_path}: {self.image.width} {self.image.height}')
        if label_path:
            self.label = read_label(self.label_path)
            if len(self.label) != 6:
                raise Exception('Invalid annotation: ', label_path)
        else:
            self.label = read_label(self.label_path)

def pil_to_tensor(img):
    return transforms.functional.to_tensor(img)

def tensor_to_pil(tensor):
    a = tensor.min()
    b = tensor.max()
    img = (tensor - a) / (b - a)
    return transforms.functional.to_pil_image(img)

def calc_mean_and_std(images):
    mean = 0
    std = 0
    to_tensor = transforms.ToTensor()
    for img in images:
        x = to_tensor(img)
        mean += x.mean()
        std += x.std()
    mean /= len(images)
    std /= len(images)
    return mean, std


if __name__ == '__main__':
    label = read_label('data/yolo/train/label/0001.txt')
