from typing import List

from PIL import Image
import numpy as np
from torchvision import transforms
from recordclass import recordclass, RecordClass



IMAGE_SIZE = 624

class Annotation(RecordClass):
    id: int
    rect: np.ndarray

    def copy(self):
        return Annotation(self.id, self.rect.copy())

class BBLabel(List[Annotation]):
    def copy(self):
        return [a.copy() for a in self]

def read_label(path):
    f = open(path, 'r')
    lines = f.readlines()
    label = []
    for line in lines:
        parted  = line.split(' ')
        id = int(parted[0])
        center_x, center_y, w, h = [float(v) for v in parted[1:]]
        label.append(Annotation(id, np.array([
            center_x - w / 2,
            center_x + w / 2,
            center_y - h / 2,
            center_y + h / 2,
        ])))
    return label


class XrayBBItem:
    def __init__(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path
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

