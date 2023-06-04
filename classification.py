import os
import re
from glob import glob
import math

import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image
from sklearn import metrics as skmetrics
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from timm.scheduler.cosine_lr import CosineLRScheduler

from endaaman.ml import BaseMLCLI, BaseTrainer, BaseTrainerConfig, Field, BaseDLArgs, Checkpoint, roc_auc_ci
from endaaman.metrics import BaseMetrics

from common import cols_clinical, cols_measure, col_target
from models import TimmModelWithFeatures, TimmModel, LinearModel
from datasets import XRDataset, XRROIDataset, FeatureDataset

J = os.path.join



class ROCMetrics(BaseMetrics):
    def calc(self, preds, gts):
        if len(preds) < 10:
            return None
        preds = preds.detach().cpu().numpy()
        gts = gts.detach().cpu().numpy()
        fpr, tpr, thresholds = skmetrics.roc_curve(gts, preds)
        auc = skmetrics.auc(fpr, tpr)
        youden_index = np.argmax(tpr - fpr)
        preds_bool = preds > thresholds[youden_index]
        acc = skmetrics.accuracy_score(preds_bool, gts)
        return auc, acc, tpr[youden_index], -fpr[youden_index]+1

def visualize_roc(trainer:BaseTrainer, ax, train_preds, train_gts, val_preds, val_gts):
    train_preds, train_gts, val_preds, val_gts = [
        v.detach().cpu().numpy().flatten() for v in (train_preds, train_gts, val_preds, val_gts)
    ]

    for t, preds, gts in (('train', train_preds, train_gts), ('val', val_preds, val_gts)):
        fpr, tpr, thresholds = skmetrics.roc_curve(gts, preds)
        auc = skmetrics.auc(fpr, tpr)
        lower, upper = roc_auc_ci(gts, preds)
        ax.plot(fpr, tpr, label=f'{t} AUC:{auc:.3f}({lower:.3f}-{upper:.3f})')
        if t == 'train':
            youden_index = np.argmax(tpr - fpr)
            threshold = thresholds[youden_index]

    ax.set_title(f'ROC (t={threshold:.2f})')
    ax.set_ylabel('Sensitivity')
    ax.set_xlabel('1 - Specificity')
    ax.legend(loc='lower right')


def dump_preds_gts(trainer:BaseTrainer, train_preds, train_gts, val_preds, val_gts):
    if not trainer.is_achieved_best():
        return
    names = ('train_preds', 'train_gts', 'val_preds', 'val_gts')
    vv = (train_preds, train_gts, val_preds, val_gts)
    for (name, v) in zip(names, vv):
        v = v.detach().cpu().numpy().flatten()
        np.save(J(trainer.out_dir, name), v)


class CommonTrainer(BaseTrainer):
    def get_metrics(self):
        return {
            'auc_acc_recall_spec': ROCMetrics(),
        }

    def get_visualizers(self):
        return {
            'roc': visualize_roc,
        }

    def get_hooks(self):
        return {
            'dump': dump_preds_gts,
        }


class ImageTrainerConfig(BaseTrainerConfig):
    model_name: str
    num_features: int
    size: int
    scheduler: str
    normalize_features: bool
    normalize_image: bool

class ImageTrainer(CommonTrainer):
    def prepare(self):
        self.criterion = nn.BCELoss()
        model = TimmModelWithFeatures(
            name=self.config.model_name,
            num_features=self.config.num_features)
        return model

    def create_scheduler(self):
        if re.match(r'^cosine.*', self.config.scheduler):
            return CosineAnnealingLR(self.optimizer, T_max=50, eta_min=self.config.lr/10)
        if self.config.scheduler == 'static':
            return None
        raise RuntimeError(f'Invalid')

    def eval(self, inputs, gts):
        if self.config.num_features > 0:
            inputs, features = inputs
            features = features.to(self.device)
        else:
            features = None
        inputs = inputs.to(self.device)
        outputs = self.model(inputs, features)
        loss = self.criterion(outputs, gts.to(self.device))
        return loss, outputs


class FeatureTrainerConfig(BaseTrainerConfig):
    num_features: int
    normalize_features: bool

class FeatureTrainer(CommonTrainer):
    def prepare(self):
        self.criterion = nn.BCELoss()
        model = LinearModel(num_features=self.config.num_features, num_classes=1)
        return model

    def eval(self, inputs, gts):
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, gts.to(self.device))
        return loss, outputs


class CLI(BaseMLCLI):
    class CommonArgs(BaseDLArgs):
        pass

    class TrainArgs(BaseDLArgs):
        num_features:int = Field(0, cli=('--num-features', '-F'))
        batch_size:int = Field(2, cli=('--batch-size', '-B'))
        epoch:int = 20
        raw_features = Field(False, cli=('--raw-features', ))

    class ImageArgs(TrainArgs):
        lr:float = 0.001
        model_name:str = Field('tf_efficientnet_b0', cli=('--model', '-m'))
        source:str = Field('full', cli=('--source', '-S'), regex='^full|roi$')
        size:int = 512
        raw_image = Field(False, cli=('--raw-image', ))
        scheduler:str = 'static'

    def run_image(self, a:ImageArgs):
        match a.source:
            case 'full':
                DS = XRDataset
            case 'roi':
                DS = XRROIDataset
            case _:
                raise RuntimeError('Invalid source:', a.source)

        print('Dataset type:', DS)
        dss = [DS(
            size=a.size,
            num_features=a.num_features,
            target=t,
            normalize_image=not a.raw_image,
            normalize_features=not a.raw_features,
        ) for t in ['train', 'test']]

        config = ImageTrainerConfig(
            model_name=a.model_name,
            num_features=a.num_features,
            batch_size=a.batch_size,
            num_workers=a.num_workers,
            lr=a.lr,
            size=a.size,
            scheduler=a.scheduler,
            normalize_image=not a.raw_image,
            normalize_features=not a.raw_features,
        )
        trainer = ImageTrainer(
            config=config,
            out_dir=f'out/classification/{a.source}_{a.num_features}/{a.model_name}',
            train_dataset=dss[0],
            val_dataset=dss[1],
            experiment_name='classification',
            main_metrics='auc',
            overwrite=a.overwrite,
        )

        trainer.start(a.epoch)


    class FeatureArgs(TrainArgs):
        lr:float = 0.0001
        model_name:str = Field('linear', cli=('--model', '-m'))

    def run_feature(self, a:FeatureArgs):
        dss = [FeatureDataset(
            num_features=self.a.num_features,
            target=t,
            normalize_features=not a.raw_features,
        ) for t in ['train', 'test']]

        config = FeatureTrainerConfig(
            num_features=a.num_features,
            batch_size=a.batch_size,
            num_workers=a.num_workers,
            lr=a.lr,
            normalize_features=not a.raw_features,
        )
        trainer = FeatureTrainer(
            config=config,
            out_dir=f'out/classification/feature_{a.num_features}/{a.model_name}',
            train_dataset=dss[0],
            val_dataset=dss[1],
            experiment_name='classification',
            main_metrics='auc',
            overwrite=a.overwrite,
        )

        trainer.start(a.epoch)

    class PredictArgs(CommonArgs):
        experiment_dir: str = Field(..., cli=('--exp-dir', '-e'))

    # def run_predict(self, a:PredictArgs):
    #     checkpoint:Checkpoint = torch.load(J(a.experiment_dir, 'checkpoint_best.pt'))
    #     print(checkpoint.config)
    #     rest_path, model_name = os.path.split(a.experiment_dir)

    #     # remove trailing suffix number
    #     model_name = re.sub('_\d*$', '', model_name, 1)

    #     rest_path, mode = os.path.split(rest_path)

    #     source, num_features = mode.split('_')
    #     num_features = int(num_features)

    #     match model_name:
    #         case 'linear':
    #             model = LinearModel(num_features, 1)
    #             config = FeatureTrainerConfig(**checkpoint.config)
    #         case _:
    #             model = TimmModelWithFeatures(model_name, num_features, 1)
    #             config = ImageTrainerConfig(**checkpoint.config)

    #     ds = XRDataset(size=config.size, num_features=num_features, target='test')

    # def arg_ds(self, parser):
    #     parser.add_argument('--checkpoint', '-c', required=True)
    #     parser.add_argument('--target', '-t', default='test', choices=['train', 'test'])

    # def run_ds(self):
    #     checkpoint = torch.load(self.args.checkpoint, map_location=lambda storage, loc: storage)
    #     predictor = self.create_predictor(P=ClsPredictor, checkpoint=checkpoint)

    #     ds = XRDataset(target=self.a.target, with_features=predictor.with_features, aug_mode='test')
    #     loader = DataLoader(dataset=ds, batch_size=self.a.batch_size, num_workers=1)

    #     results = predictor.predict(loader=loader)

    #     pred_y = np.array(results)
    #     true_y = np.array([i.treatment for i in ds.items])
    #     print(metrics.roc_auc_score(true_y, pred_y))

    #     print('done')


if __name__ == '__main__':
    cli = CLI()
    cli.run()
