import re
import torch
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
    def __init__(self, name, with_features, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.with_features = with_features
        self.base = timm.create_model(name, pretrained=True, in_chans=1, num_classes=num_classes)
        cnn_num_features = self.base.num_features

        S = 128
        # self.fc_image = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     nn.Linear(
        #         in_features=self.base.num_features,
        #         out_features=S,
        #     ),
        # )
        # cnn_num_features = S

        self.fc_feature = nn.Sequential(
            nn.Linear(
                in_features=8,
                out_features=S,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=S,
                out_features=S,
            ),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(
            in_features=cnn_num_features + S,
            out_features=1
        )

    def get_cam_layer(self):
        return self.base.conv_head

    def do_activate(self, x, activate):
        if not activate:
            return x
        if self.num_classes > 1:
            return torch.softmax(x, dim=1)
        return torch.sigmoid(x)

    def forward(self, x, features, activate=True):
        if self.with_features == 0:
            x = self.base(x)
            return self.do_activate(x, activate)

        x = self.base.forward_features(x)
        x = self.base.forward_head(x, pre_logits=True)
        # x = self.fc_image(x)

        features = self.fc_feature(features)

        # original
        x = torch.cat([x, features], dim=1)
        x = self.fc(x)

        return self.do_activate(x, activate)


class Dense(nn.Module):
    def __init__(self, a, b=None):
        super().__init__()
        if not b:
            b = a
        self.dense = nn.Linear(a, b)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dense(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x


class LinearModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        cfg = [
            # 128,
            64,
            64,
            num_classes,
        ]
        last_feat = num_features
        layers = []
        for i, n in enumerate(cfg):
            layers.append(nn.Linear(last_feat, n))
            last_feat = n
            if i != len(cfg) - 1:
                layers += [
                    # nn.BatchNorm1d(n),
                    # nn.Dropout(p=0.2),
                    nn.ReLU(inplace=True),
                ]

        self.fc = nn.Sequential(*layers)

    def forward(self, x, activate=True):
        x = self.fc(x)
        if activate:
            if self.num_classes > 1:
                x = torch.softmax(x, dim=1)
            else:
                x = torch.sigmoid(x)
        return x
