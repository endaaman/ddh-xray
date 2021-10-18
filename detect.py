import os
import json
import yaml

from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import albumentations as A
# from effdet import EfficientDet, FocalLoss, EFFDET_PARAMS
from effdet import EfficientDet, DetBenchTrain, get_efficientdet_config
from effdet.efficientdet import HeadNet
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
import pytorch_lightning as pl

from datasets import EffdetDataset, YOLODataset
from utils import get_state_dict
from models import Darknet
from endaaman import TorchCommander


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

class BaseModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.lr = args.lr
        self.log_every_n_steps = 12

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)

    def train_dataloader(self):
        assert self.image_size
        assert hasattr(self, 'train_dataset')
        ds = self.train_dataset()
        augs = [] if self.args.no_aug else [
            A.RandomResizedCrop(width=self.image_size, height=self.image_size, scale=[0.7, 1.0]),
            A.HorizontalFlip(p=0.5),
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.MotionBlur(p=.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=5, p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
        ]
        ds.apply_augs(augs)
        return DataLoader(
            ds,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )


class EffdetModule(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = get_efficientdet_config(f'tf_efficientdet_{self.args.depth}')
        self.model = EfficientDet(cfg)
        self.bench = DetBenchTrain(self.model)
        self.image_size = SIZE_BY_DEPTH[self.args.depth]

    def train_dataset(self):
        return EffdetDataset()

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.bench(x, y)
        return {'loss': loss['loss']}


class YOLOModule(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = Darknet()
        self.image_size = 512

    def train_dataset(self):
        return YOLODataset()

    def training_step(self, batch, batch_idx):
        x, targets = batch
        for idx, tt in enumerate(targets):
            tt[:, 0] = idx
        targets = targets.view(-1, 6) # batch x [batch_idx, cls_id, x, y, w, h]
        print(targets)
        loss, y = self.model(x, targets)
        return {'loss': loss}


class MyTrainer(TorchCommander):
    def arg_common(self, parser):
        parser.add_argument('-e', '--epoch', type=int, default=50)
        parser.add_argument('-b', '--batch-size', type=int, default=16)
        parser.add_argument('--lr', type=float, default=0.01)
        parser.add_argument('--no-aug', action='store_true')
        parser.add_argument('--workers', type=int, default=os.cpu_count()//2)

    def pre_common(self):
        self.trainer = pl.Trainer(log_every_n_steps=16)

    def arg_effdet(self, parser):
        parser.add_argument('-d', '--depth', default='d0', type=str, choices=list(SIZE_BY_DEPTH.keys()))

    def run_effdet(self):
        m = EffdetModule(self.args)
        self.trainer.fit(m)

    def run_yolo(self):
        m = YOLOModule(self.args)
        self.trainer.fit(m)

MyTrainer().run()
