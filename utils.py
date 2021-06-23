import numpy as np
from torchvision import transforms
from recordclass import recordclass, RecordClass


class Annotation(RecordClass):
    id: int
    rect: np.ndarray

def pil_to_tensor(img):
    return transforms.functional.to_tensor(img)

def tensor_to_pil(tensor):
    a = tensor.min()
    b = tensor.max()
    img = (tensor - a) / (b - a)
    return transforms.functional.to_pil_image(img)
