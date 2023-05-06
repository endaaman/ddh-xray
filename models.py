import torch
import re
import timm
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, models
from effdet import EfficientDet, get_efficientdet_config

# from .ssd import SSD300
# from .yolo_v3 import YOLOv3
# from .yolor import Yolor, YOLOv4, yolor_loss


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
    def __init__(self, name='tf_efficientnetv2_b0', num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.base = timm.create_model(name, pretrained=True, in_chans=1, num_classes=num_classes)

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
    def __init__(self, name, num_features, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.base = timm.create_model(name, pretrained=True, in_chans=1, num_classes=num_classes)
        self.fc = nn.Linear(
            in_features=self.base.classifier.in_features + num_features,
            out_features=num_classes
        )

    def get_cam_layer(self):
        return self.base.conv_head

    def forward(self, x, features, activate=True):
        x = self.base.forward_features(x)
        x = self.base.forward_head(x, pre_logits=True)
        if self.num_features > 0 and features is not None:
            x = torch.cat([x, features], dim=1)
        x = self.fc(x)
        if activate:
            if self.num_classes > 1:
                x = torch.softmax(x, dim=1)
            else:
                x = torch.sigmoid(x)
        return x
