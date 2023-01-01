import os
import sys
import io
import json
import yaml

from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np
import matplotlib
from matplotlib import ticker, pyplot as plt
import torch
from torch import nn
from torch import optim
from torchvision import transforms
# from torch.utils.tensorboard import SummaryWriter
from effdet import DetBenchTrain
from ptflops import get_model_complexity_info
from timm.scheduler import CosineLRScheduler

from endaaman.trainer import TrainCommander, Trainer

from datasets import XRBBDataset
from utils import get_state_dict
from models import create_det_model, yolor_loss
from models.ssd import MultiBoxLoss


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


class EffDetTrainer(Trainer):
    def prepare(self, **kwargs):
        # self.criterion = FocalBCELoss(gamma=4.0)
        self.bench = DetBenchTrain(self.model).to(self.device)

    # def create_scheduler(self, lr):
    #     return CosineLRScheduler(
    #         self.optimizer, t_initial=100, lr_min=0.00001,
    #         warmup_t=10, warmup_lr_init=0.00005, warmup_prefix=True)
    # def hook_load_state(self, checkpoint):
    #     self.scheduler.step(checkpoint.epoch-1)
    # def step(self, train_loss):
    #     self.scheduler.step(self.current_epoch)

    def eval(self, inputs, labels):
        inputs = inputs.to(self.device)
        labels['bbox'] = labels['bbox'].to(self.device)
        labels['cls'] = labels['cls'].to(self.device)
        loss = self.bench(inputs, labels)
        return loss['loss'], None

    def get_metrics(self):
        return {
            'batch': {
            },
            'epoch': {
            },
        }



class CMD(TrainCommander):
    def arg_common(self, parser):
        pass
        # parser.add_argument('--no-aug', action='store_true')

    def create_loaders(self, mode, image_size, collate_fn=None):
        if mode not in ['effdet', 'yolo', 'ssd']:
            raise ValueError(f'Invalid target: {mode}')

        return [
            self.as_loader(
                XRBBDataset(mode=mode, target=target, size=image_size),
                collate_fn=collate_fn
            ) for target in ['train', 'test']
        ]

    def arg_effdet(self, parser):
        parser.add_argument('-d', '--depth', default='d0', choices=list(SIZE_BY_DEPTH.keys()))

    def run_effdet(self):
        name = f'effdet_{self.a.depth}'
        model = create_det_model(name)
        loaders = self.create_loaders('effdet', SIZE_BY_DEPTH[self.args.depth])

        trainer = self.create_trainer(
            T=EffDetTrainer,
            name=name,
            model=model,
            loaders=loaders,
        )

        self.start(trainer)





if __name__ == '__main__':
    cmd = CMD({
        'epoch': 100,
        'lr': 0.0001,
        'batch_size': 32,
    })
    cmd.run()
