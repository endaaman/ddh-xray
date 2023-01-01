import os
import re
from glob import glob

from tqdm import tqdm
import pandas as pd
from PIL import Image
from sklearn import metrics
import numpy as np
import torch
from torch import nn
from torch import optim
from timm.scheduler.cosine_lr import CosineLRScheduler
from endaaman.trainer import TrainCommander, Trainer
from endaaman.metrics import BinaryAccuracy, BinaryAUC, BinaryRecall, BinarySpecificity

from models import create_model
from datasets import XRDataset, XRROIDataset


class T(Trainer):
    def prepare(self, **kwargs):
        # self.criterion = FocalBCELoss(gamma=4.0)
        self.criterion = nn.BCELoss()

    def create_scheduler(self, lr):
        return CosineLRScheduler(
            self.optimizer, t_initial=100, lr_min=0.00001,
            warmup_t=10, warmup_lr_init=0.00005, warmup_prefix=True)

    def hook_load_state(self, checkpoint):
        self.scheduler.step(checkpoint.epoch-1)

    def step(self, train_loss):
        self.scheduler.step(self.current_epoch)

    def eval(self, inputs, labels):
        outputs = self.model(inputs.to(self.device))
        loss = self.criterion(outputs, labels.to(self.device))
        return loss, outputs

    def get_metrics(self):
        return {
            'batch': {
                'acc': BinaryAccuracy(),
                'recall': BinaryRecall(),
                'spec': BinarySpecificity()
            },
            'epoch': {
                'auc': BinaryAUC(),
            },
        }


class CMD(TrainCommander):
    def arg_common(self, parser):
        parser.add_argument('--model', '-m', default='tf_efficientnetv2_b0')

    def arg_xr(self, parser):
        parser.add_argument('--size', type=int, default=768)

    def run_xr(self):
        model = create_model(self.args.model)

        loaders = [self.as_loader(XRDataset(
            size=self.args.size,
            target=t,
        )) for t in ['train', 'test']]

        trainer = self.create_trainer(
            T=T,
            name='xr_' + self.args.model,
            model=model,
            loaders=loaders,
        )

        trainer.start(self.args.epoch, lr=self.args.lr)


    def arg_roi(self, parser):
        parser.add_argument('--size', type=int, default=512)

    def run_roi(self):
        model = create_model(self.args.model)

        loaders = [self.as_loader(XRROIDataset(
            size=self.args.size,
            target=t,
        )) for t in ['train', 'test']]

        trainer = self.create_trainer(
            T=T,
            name='roi_' + self.args.model,
            model=model,
            loaders=loaders,
        )

        trainer.start(self.args.epoch, lr=self.args.lr)


if __name__ == '__main__':
    cmd = CMD({
        'epoch': 50,
        'lr': 0.0001,
        'batch_size': 32,
    })
    cmd.run()
