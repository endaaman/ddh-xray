import os
import re
from glob import glob

from tqdm import tqdm
import pandas as pd
from PIL import Image
from sklearn import metrics
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from timm.scheduler.cosine_lr import CosineLRScheduler
from endaaman.torch import TorchCommander, Trainer, Predictor
from endaaman.metrics import BinaryAccuracy, BinaryAUC, BinaryRecall, BinarySpecificity

from models import create_model, TimmModelWithFeatures
from datasets import XRDataset, XRROIDataset


class MyTrainer(Trainer):
    def prepare(self, **kwargs):
        # self.criterion = FocalBCELoss(gamma=4.0)
        self.criterion = nn.BCELoss()
        self.with_features = kwargs.get('with_features', False)
        return create_model(self.model_name)

    def create_scheduler(self, lr):
        return CosineLRScheduler(
            self.optimizer,
            warmup_t=5, t_initial=70,
            warmup_lr_init=lr/2, lr_min=lr/100,
            warmup_prefix=True)

    def hook_load_state(self, checkpoint):
        self.scheduler.step(checkpoint.epoch-1)

    def step(self, train_loss):
        self.scheduler.step(self.current_epoch)

    def eval(self, inputs, labels):
        if self.with_features:
            inputs, features = inputs
            features.requires_grad = True
            outputs = self.model(inputs.to(self.device), features.to(self.device))
        else:
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


class MyPredictor(Predictor):
    def prepare(self, **kwargs):
        self.with_features = isinstance(self.model, TimmModelWithFeatures)

    def create_model(self):
        model = create_model(self.checkpoint.model_name)
        model.load_state_dict(self.checkpoint.model_state)
        return model.to(self.device).eval()

    def eval(self, inputs):
        if self.with_features:
            inputs, features = inputs
            return self.model(inputs.to(self.device), features.to(self.device)).detach().cpu()
        return self.model(inputs.to(self.device)).detach().cpu()

    def collate(self, pred, idx):
        return pred.item()


class CMD(TorchCommander):
    def arg_common(self, parser):
        parser.add_argument('--model', '-m', default='tf_efficientnetv2_b0')

    def arg_xr(self, parser):
        parser.add_argument('--size', type=int, default=768)
        parser.add_argument('--features', '-f', type=int, default=0)

    def run_xr(self):
        name = self.args.model
        with_features = False
        if self.a.features == 1:
            name = f'{name}_f'
            with_features = True
        elif self.a.features == 2:
            name = f'{name}_f2'
            with_features = True
        else:
            raise RuntimeError(f'Invalid --features: {self.a.features}')

        loaders = [self.as_loader(XRDataset(
            size=self.args.size,
            target=t,
            with_features=with_features,
        )) for t in ['train', 'test']]

        trainer = self.create_trainer(
            T=MyTrainer,
            model_name=name,
            loaders=loaders,
            trainer_name=f'xr_{name}',
            log_dir='data/logs_xr',
            with_features=with_features,
        )

        trainer.start(self.args.epoch, lr=self.args.lr)


    # def arg_roi(self, parser):
    #     parser.add_argument('--size', type=int, default=512)

    def run_roi(self):
        loaders = [self.as_loader(XRROIDataset(
            size=(512, 256),
            target=t,
            with_features=self.a.with_features,
        )) for t in ['train', 'test']]

        trainer = self.create_trainer(
            T=MyTrainer,
            model_name=self.args.model,
            loaders=loaders,
            trainer_name='roi_' + self.args.model,
            log_dir='data/logs_roi',
            with_features=self.a.with_features
        )

        trainer.start(self.args.epoch, lr=self.args.lr)


    def arg_ds(self, parser):
        parser.add_argument('--checkpoint', '-c', required=True)
        parser.add_argument('--target', '-t', default='test', choices=['train', 'test'])

    def run_ds(self):
        checkpoint = torch.load(self.args.checkpoint, map_location=lambda storage, loc: storage)
        predictor = self.create_predictor(P=MyPredictor, checkpoint=checkpoint)

        ds = XRDataset(target=self.a.target, size=768, with_features=predictor.with_features, aug_mode='test')
        loader =DataLoader(dataset=ds, batch_size=self.a.batch_size, num_workers=1)

        results = predictor.predict(loader=loader)

        pred_y = np.array(results)
        true_y = np.array([i.treatment for i in ds.items])
        print(metrics.roc_auc_score(true_y, pred_y))

        print('done')


if __name__ == '__main__':
    cmd = CMD({
        'epoch': 75,
        'lr': 0.00002,
        'batch_size': 16,
    })
    cmd.run()
