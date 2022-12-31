# ported from: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
import torch
import re
import timm
from torch import nn
from torchvision import transforms, models

from .ssd import SSD300
from .yolo_v3 import YOLOv3
from .yolor import Yolor, YOLOv4, yolor_loss


class TimmModel(nn.Module):
    def __init__(self, name='tf_efficientnetv2_b0', num_classes=1, activation=True):
        super().__init__()
        self.num_classes = num_classes
        self.activation = activation
        self.base = timm.create_model(name, pretrained=True, num_classes=num_classes)

    def get_cam_layer(self):
        return self.base.conv_head

    def forward(self, x, activate=True):
        x = self.base(x)
        if activate:
            if self.num_classes > 1:
                x = torch.softmax(x, dim=1)
            else:
                x = torch.sigmoid(x)
        return x


def create_model(s):
    return TimmModel(name=s, num_classes=1)

def create_det_model(s):
    return TimmModel(name=s, num_classes=1)
