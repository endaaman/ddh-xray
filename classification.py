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
        self.lr_min = kwargs.pop('lr_min')
        self.lr_initial = kwargs.pop('lr_initial')
        assert len(kwargs) == 0
        self.criterion = nn.BCELoss()
        model =  create_model(self.model_name)
        self.with_features = isinstance(model, TimmModelWithFeatures)
        return model

    def create_scheduler(self, total_epoch):
        return CosineLRScheduler(
            self.optimizer,
            warmup_t=0, t_initial=self.lr_initial,
            warmup_lr_init=self.lr/2, lr_min=self.lr/self.lr_min,
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


class BasePredictor(Predictor):
    def prepare(self, **kwargs):
        model = create_model(self.checkpoint.model_name)
        model.load_state_dict(self.checkpoint.model_state)
        self.with_features = isinstance(model, TimmModelWithFeatures)
        return model.to(self.device).eval()

class ClsPredictor(BasePredictor):
    def eval(self, inputs):
        if self.with_features:
            inputs, features = inputs
            return self.model(inputs.to(self.device), features.to(self.device)).detach().cpu()
        return self.model(inputs.to(self.device)).detach().cpu()

    def collate(self, pred, idx):
        return pred.item()

class FeaturePredictor(BasePredictor):
    def eval(self, inputs):
        m = self.model.base

        x = inputs.to(self.device)
        x = m.forward_features(x)
        x = m.global_pool(x)
        return x.detach().cpu()

    def collate(self, pred, idx):
        return pred


class CMD(TorchCommander):
    def arg_common(self, parser):
        parser.add_argument('--model', '-m', default='tf_efficientnetv2_b0')
        parser.add_argument('--features', '-f', type=int, default=0, choices=[0, 1, 2])
        parser.add_argument('--test-ratio', '-r', type=float, default=-1)
        parser.add_argument('--lr-min', type=int, default=100)
        parser.add_argument('--lr-initial', type=int, default=30)

    def arg_xr(self, parser):
        parser.add_argument('--size', type=int, default=512)

    def get_model_name(self):
        name = self.args.model
        if  self.a.features > 0:
            name = f'{name}_f{self.a.features}'
        return name

    def run_xr(self):
        loaders = self.as_loaders(*[XRDataset(
            size=self.args.size,
            with_features=self.a.features > 0,
            test_ratio=self.a.test_ratio,
            target=t,
        ) for t in ['train', 'test']])

        name = self.get_model_name()
        trainer = self.create_trainer(
            T=MyTrainer,
            model_name=name,
            loaders=loaders,
            trainer_name=f'xr_{name}',
            lr_min=self.a.lr_min,
            lr_initial=self.a.lr_initial,
        )

        trainer.start(self.args.epoch)


    def arg_roi(self, parser):
        parser.add_argument('--size', type=int, default=512)
        parser.add_argument('--base-dir', '-d', default='data/rois/gt')

    def run_roi(self):
        loaders = self.as_loaders(*[XRROIDataset(
            base_dir=self.a.base_dir,
            with_features=self.a.features > 0,
            size=(self.a.size, self.a.size//2),
            target=t,
        ) for t in ['train', 'test']])

        name = self.get_model_name()
        trainer = self.create_trainer(
            T=MyTrainer,
            model_name=name,
            loaders=loaders,
            trainer_name='roi_' + name,
            lr_min=self.a.lr_min,
        )

        trainer.start(self.args.epoch)

    def arg_predict_features(self, parser):
        parser.add_argument('--checkpoint', '-c', required=True)
        parser.add_argument('--target', '-t', default='test', choices=['train', 'test'])

    def run_predict_features(self):
        checkpoint = torch.load(self.args.checkpoint, map_location=lambda storage, loc: storage)
        predictor = self.create_predictor(P=FeaturePredictor, checkpoint=checkpoint)
        ds = XRDataset(target=self.a.target, with_features=predictor.with_features, aug_mode='test')
        loader = DataLoader(dataset=ds, batch_size=self.a.batch_size, num_workers=1)
        results = predictor.predict(loader=loader)
        results = torch.stack(results)

        p = os.path.join(predictor.get_out_dir(), f'features_{self.a.target}.pt')
        torch.save(results, p)
        print(f'wrote {p}')

    def arg_ds(self, parser):
        parser.add_argument('--checkpoint', '-c', required=True)
        parser.add_argument('--target', '-t', default='test', choices=['train', 'test'])

    def run_ds(self):
        checkpoint = torch.load(self.args.checkpoint, map_location=lambda storage, loc: storage)
        predictor = self.create_predictor(P=ClsPredictor, checkpoint=checkpoint)

        ds = XRDataset(target=self.a.target, with_features=predictor.with_features, aug_mode='test')
        loader = DataLoader(dataset=ds, batch_size=self.a.batch_size, num_workers=1)

        results = predictor.predict(loader=loader)

        pred_y = np.array(results)
        true_y = np.array([i.treatment for i in ds.items])
        print(metrics.roc_auc_score(true_y, pred_y))

        print('done')


if __name__ == '__main__':
    cmd = CMD({
        'epoch': 50,
        'lr': 0.0001,
        'batch_size': 4,
    })
    cmd.run()
