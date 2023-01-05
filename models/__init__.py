import torch
import re
import timm
from torch import nn
from torchvision import transforms, models
from effdet import EfficientDet, get_efficientdet_config
import torch.nn.functional as F

from .ssd import SSD300
from .yolo_v3 import YOLOv3
from .yolor import Yolor, YOLOv4, yolor_loss


SIZE_BY_DEPTH = {
    'd0': 128 * 4,
    'd1': 128 * 5,
    'd2': 128 * 6,
    'd3': 128 * 7,
    'd4': 128 * 8,
    'd5': 128 * 10,
    'd6': 128 * 12,
    'd7': 128 * 14,
}

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


class TimmModelWithFeatures(nn.Module):
    def __init__(self, name='tf_efficientnetv2_b0', num_measurement=10, num_classes=1, activation=True):
        super().__init__()
        self.num_classes = num_classes
        self.activation = activation
        self.base = timm.create_model(name, pretrained=True, num_classes=num_classes)
        self.classifier = nn.Linear(self.base.classifier.in_features+num_measurement, num_classes)

    def get_cam_layer(self):
        return self.base.conv_head

    def forward(self, x, measurement, activate=True):
        x = self.base.forward_features(x)

        x = self.base.global_pool(x)
        if self.base.drop_rate > 0.:
            x = F.dropout(x, p=self.base.drop_rate, training=self.base.training)

        x = torch.cat([x, measurement], dim=1)
        x = self.classifier(x)

        if activate:
            if self.num_classes > 1:
                x = torch.softmax(x, dim=1)
            else:
                x = torch.sigmoid(x)
        return x

def create_model(name):
    if m := re.match(r'(^.*)_f$', name):
        return TimmModelWithFeatures(m[1])
    return TimmModel(name)


def create_det_model(name):
    if name == 'yolo':
        return YOLOv3()

    if name == 'yolor':
        return Yolor(num_classes=7)

    # if name == 'yolo5':
    #     # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    #     return Yolo5(cfg='cfg/yolov5s.yaml')

    if m := re.match(r'^effdet_d(\d)$', name):
        cfg = get_efficientdet_config(f'tf_efficientdet_d{m[1]}')
        cfg.num_classes = 6
        return EfficientDet(cfg)

    if name == 'ssd':
        return SSD300(n_classes=7)

    raise ValueError(f'Ivalid name: {name}')


